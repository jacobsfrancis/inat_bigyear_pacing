"""
iNat Pace Dashboard (CS101 edition)

What this app does:
1) You type:
   - YOUR iNaturalist username (login)
   - one or more "friend" usernames

2) The app pulls data from the iNaturalist API:
   - Your daily observation counts for THIS YEAR (YTD)
   - For each friend: their observation counts by year, picks their best year,
     then pulls daily counts for that best year

3) It converts daily counts -> cumulative totals across day-of-year (DOY)
4) It plots: your current-year cumulative curve vs each friend's best-year curve
5) It prints a small table: "how far ahead/behind" you are today vs each friend

Important concept:
- The API endpoint we use is /v1/observations/histogram
  It returns "binned" counts (per day, per year, etc.) without downloading every observation.
"""

import datetime as dt
import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Base URL for iNaturalist v1 API
BASE = "https://api.inaturalist.org/v1"


# =========================================================
# 1) Small helpers to deal with messy / varying JSON shapes
# =========================================================

def coerce_histogram_results(raw):
    """
    Convert histogram 'results' into a plain dict {str_key: int_count}.

    Why?
    APIs sometimes return:
      - {"2026-01-01": 10, "2026-01-02": 5}
    or:
      - {"2026-01-01": {"count": 10}, ...}
    or:
      - [{"date": "2026-01-01", "count": 10}, ...]

    This function tries to normalize all those to:
      {"2026-01-01": 10, ...}
    """
    out = {}

    # Case A: list of dicts
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            key = item.get("date") or item.get("day") or item.get("year")
            val = item.get("count") or item.get("total") or item.get("value")
            if key is not None and val is not None:
                out[str(key)] = int(val)
        return out

    # Case B: dict
    if isinstance(raw, dict):
        for k, v in raw.items():
            # nested dict values like {"count": 10}
            if isinstance(v, dict):
                if "count" in v:
                    out[str(k)] = int(v["count"])
                elif "total" in v:
                    out[str(k)] = int(v["total"])
                elif "value" in v:
                    out[str(k)] = int(v["value"])
                # else: unknown shape; skip
            else:
                # direct integer / numeric values
                try:
                    out[str(k)] = int(v)
                except Exception:
                    continue
        return out

    # Unknown shape
    return out


def debug_block(title, response):
    """
    Print useful debugging information in the Streamlit UI.
    This is how you learn what the API is returning.
    """
    st.subheader(f"DEBUG: {title}")
    st.write("URL:", response.url)
    st.write("Status code:", response.status_code)
    try:
        st.write("Raw JSON:", response.json())
    except Exception as e:
        st.write("JSON decode error:", str(e))


# =========================================================
# 2) API calls: histogram by YEAR and by DAY
# =========================================================

def histogram_year_counts(user_identifier: str, debug: bool = False) -> dict[int, int]:
    """
    Ask iNat: "How many observations did this user make each year?"

    Endpoint:
      GET /v1/observations/histogram

    Parameters:
      user_id    = username OR numeric id
      interval   = "year"
      date_field = "observed"  (use observed_on date, not upload date)

    Returns:
      {year_int: count_int}
      e.g. {2024: 1709, 2025: 3826, ...}

    IMPORTANT QUIRK (from your debug output):
      The API may return:
        "results": {"year": {"2025-01-01": 3826, ...}}
      so keys are "YYYY-01-01" not "YYYY".
      We parse the first 4 characters as the year.
    """
    url = f"{BASE}/observations/histogram"
    params = {
        "user_id": user_identifier,
        "interval": "year",
        "date_field": "observed",
    }

    r = requests.get(url, params=params, timeout=30)

    if debug:
        debug_block("histogram_year_counts()", r)

    r.raise_for_status()

    # Pull out the "results" object from the JSON response
    raw = r.json().get("results", {})

    # iNat can nest the dictionary we want inside results["year"]
    if isinstance(raw, dict) and "year" in raw and isinstance(raw["year"], dict):
        raw = raw["year"]

    # Now raw should be a dict like {"2025-01-01": 3826, ...}
    # Convert to {2025: 3826, ...}
    out = {}
    for k, v in raw.items():
        try:
            year = int(str(k)[:4])  # "2025-01-01" -> 2025
            out[year] = int(v)
        except Exception:
            continue

    return out


def histogram_daily_counts(user_identifier: str, year: int, debug: bool = False) -> pd.Series:
    """
    Ask iNat: "How many observations did this user make each day in a given year?"

    Endpoint:
      GET /v1/observations/histogram

    Parameters:
      user_id    = username OR numeric id
      interval   = "day"
      date_field = "observed"
      d1/d2      = date range

    Returns:
      pandas Series indexed by date with daily counts:
        2026-01-01    139
        2026-01-02     15
        ...

    IMPORTANT QUIRK (from your debug output):
      The API may return:
        "results": {"day": {"2026-01-01": 139, ...}}
      so we unwrap results["day"].
    """
    url = f"{BASE}/observations/histogram"
    params = {
        "user_id": user_identifier,
        "interval": "day",
        "date_field": "observed",
        # Using slash format matches what often appears in iNat examples
        "d1": f"{year}/01/01",
        "d2": f"{year}/12/31",
    }

    r = requests.get(url, params=params, timeout=30)

    if debug:
        debug_block("histogram_daily_counts()", r)

    r.raise_for_status()

    raw = r.json().get("results", {})

    # iNat can nest the dictionary we want inside results["day"]
    if isinstance(raw, dict) and "day" in raw and isinstance(raw["day"], dict):
        raw = raw["day"]

    # Normalize weird shapes (nested dicts, lists, etc.)
    raw = coerce_histogram_results(raw)

    # If nothing came back, return an empty Series
    if not raw:
        return pd.Series(dtype="int64")

    # Convert dict -> Series, and parse index as dates
    s = pd.Series(raw, dtype="int64")
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


# =========================================================
# 3) Convert daily counts -> cumulative curve by day-of-year
# =========================================================

def daily_to_cumulative_by_doy(daily: pd.Series, ytd_only: bool) -> pd.DataFrame:
    """
    Input:
      daily = Series indexed by date with counts per day

    Output:
      DataFrame with columns:
        - doy (day-of-year: 1..365/366)
        - cumulative (running total up to that day)

    Key steps:
      1) Reindex to ALL days of the year so missing days become 0.
         (Histogram might omit days, or include them as 0; either way, we standardize.)
      2) Optionally cut off at "today" for year-to-date.
      3) Cumulative sum.
    """
    if daily is None or len(daily) == 0:
        return pd.DataFrame({"doy": [], "cumulative": []})

    year = int(daily.index[0].year)

    # Full daily date range for that year
    full_range = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")

    # Fill missing dates with 0
    daily = daily.reindex(full_range, fill_value=0)

    # If ytd_only, keep only up through today
    if ytd_only:
        today = pd.Timestamp.today().normalize()
        daily = daily[daily.index <= today]

    return pd.DataFrame({
        "doy": daily.index.dayofyear,
        "cumulative": daily.cumsum()
    })


# =========================================================
# 4) Streamlit UI (the "dashboard" part)
# =========================================================

st.set_page_config(page_title="iNat Pace Dashboard", layout="wide")
st.title("iNaturalist Pace Dashboard")

# Sidebar controls: user input + debug toggle
with st.expander("Inputs", expanded=True):
    me_user = st.text_input(
        "Your iNat username (login)",
        value="your_username_here"
    )

    friends_raw = st.text_input(
        "friend usernames (comma-separated)",
        value="friend1_username,friend2_username"
    )

    # If checked, the app prints URL + raw JSON for every API call
    debug = st.checkbox("Debug API responses")

    st.caption("This version plots observation pacing only (no family diversity yet).")

# Parse friend list from comma-separated string
friends = [x.strip() for x in friends_raw.split(",") if x.strip()]

# Main "Run" button
if st.button("Run"):
    try:
        # Current calendar year
        this_year = dt.date.today().year

        # -------------------------
        # YOUR curve (this year YTD)
        # -------------------------
        me_daily = histogram_daily_counts(me_user, this_year, debug=debug)
        me_curve = daily_to_cumulative_by_doy(me_daily, ytd_only=True)

        # If we still got nothing, show a clear error
        if len(me_curve) == 0:
            st.error(
                f"No daily histogram data returned for '{me_user}' in {this_year}. "
                "Double-check spelling of the username, and use Debug to inspect the API response."
            )
            st.stop()

        # Set up the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot YOUR curve as a bold dashed line
        ax.plot(
            me_curve["doy"],
            me_curve["cumulative"],
            linestyle="--",
            linewidth=3,
            label=f"{me_user} {this_year} (YTD)"
        )

        # A table of "how far ahead/behind" comparisons
        rows = []

        # Today's day-of-year for snapshot comparisons
        today_doy = int(pd.Timestamp.today().dayofyear)

        # Your cumulative total today (last point on your YTD curve)
        me_today = int(me_curve["cumulative"].iloc[-1])

        # -----------------------------------------
        # For each friend:
        # 1) find their best year
        # 2) fetch daily counts for that year
        # 3) plot their best-year curve
        # 4) compute snapshot difference at today_doy
        # -----------------------------------------
        for opp in friends:
            # 1) Per-year totals
            year_counts = histogram_year_counts(opp, debug=debug)
            if not year_counts:
                st.warning(f"No yearly histogram data returned for friend '{opp}'. Skipping.")
                continue

            # Optionally ignore current year if you only want *completed* years:
            # year_counts.pop(this_year, None)

            # 2) Choose best year = year with maximum total
            best_year = max(year_counts, key=year_counts.get)
            best_total = year_counts[best_year]

            # 3) Daily counts for that best year, then cumulative curve
            opp_daily = histogram_daily_counts(opp, best_year, debug=debug)
            opp_curve = daily_to_cumulative_by_doy(opp_daily, ytd_only=False)

            if len(opp_curve) == 0:
                st.warning(f"No daily data returned for friend '{opp}' in best year {best_year}. Skipping plot.")
                continue

            # Plot friend curve as a solid line
            ax.plot(
                opp_curve["doy"],
                opp_curve["cumulative"],
                linewidth=2,
                label=f"{opp} best year {best_year} ({best_total})"
            )

            # 4) Snapshot: friend cumulative at today's DOY
            # We do this safely (no .iloc[0] unless we confirmed it exists).
            match = opp_curve.loc[opp_curve["doy"] == today_doy, "cumulative"]
            if len(match) > 0:
                opp_today = int(match.iloc[0])
            else:
                # fallback: use closest earlier day if exact DOY isn't present
                earlier = opp_curve.loc[opp_curve["doy"] <= today_doy, "cumulative"]
                opp_today = int(earlier.iloc[-1]) if len(earlier) > 0 else 0

            # Add a row to the summary table
            rows.append({
                "friend": opp,
                "friend best year": best_year,
                "Your obs (YTD)": me_today,
                "friend obs by same DOY": opp_today,
                "Difference (you - them)": me_today - opp_today,
            })

        # Label the plot nicely
        ax.set_xlabel("Day of year")
        ax.set_ylabel("Cumulative observations")
        ax.set_title("Pace: your current year vs opponents' best years")
        ax.legend()

        # Show plot in Streamlit
        st.pyplot(fig)

        # Show snapshot table
        st.subheader("Snapshot (today)")
        if rows:
            st.dataframe(pd.DataFrame(rows))
        else:
            st.write("No friend curves plotted (see warnings above).")

    except Exception as e:
        # Catch any exception and show it in the UI
        st.error(str(e))