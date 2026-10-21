"""Streamlit demo: pick two NYC taxi zones and a pickup time, get a trip-duration estimate with an 80% range.

Run: streamlit run app/streamlit_app.py
Uses models/app_models.joblib from scripts/train_app_models.py (or the --quick variant for development);
if neither exists, downloads the published models from the GitHub release.
"""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from taxi import data, features, intervals

st.set_page_config(page_title="NYC Taxi Trip Duration", page_icon="🚕", layout="centered")


MODELS_DIR = ROOT / "models"
# trained models are too large for git; a hosted copy of the app fetches them from the GitHub release once
MODELS_URL = "https://github.com/Venius24/ML-DATA/releases/download/app-models-v2/app_models.joblib"


@st.cache_resource(show_spinner="Loading models...")
def load_bundle():
    full = MODELS_DIR / "app_models.joblib"
    if not full.exists() and not (MODELS_DIR / "app_models_quick.joblib").exists():
        MODELS_DIR.mkdir(exist_ok=True)
        tmp = full.with_suffix(".part")
        try:
            urllib.request.urlretrieve(MODELS_URL, tmp)
            tmp.rename(full)
        except OSError:
            tmp.unlink(missing_ok=True)
    for name, quick in [("app_models.joblib", False), ("app_models_quick.joblib", True)]:
        path = MODELS_DIR / name
        if path.exists():
            return joblib.load(path), quick
    return None, None


bundle, quick = load_bundle()
if bundle is None:
    st.error("No trained models found. Run `python scripts/train_app_models.py` first.")
    st.stop()

zones = bundle["zones"]
known = zones[zones["x"].notna() & ~zones["Borough"].isin(["Unknown", "N/A"])]
BOROUGHS = sorted(known["Borough"].unique())


def predict(pu: int, do: int, pickups, passengers: int):
    """Point estimate and calibrated 80% interval, in minutes, for one route at several pickup times."""
    trips = pd.DataFrame({"pickup": pd.to_datetime(pickups), "vendor": 2, "passenger_count": float(passengers),
                          "pu": pu, "do": do})
    X = features.build_features(trips, zones)
    point = bundle["point"].predict(X)
    lo, hi = intervals.apply_correction(bundle["lo"].predict(X), bundle["hi"].predict(X), bundle["cqr_shift_s"],
                                        floor=data.MIN_DURATION_S)
    return point / 60, lo / 60, hi / 60, X["dist_ft"].to_numpy() / 5280


def zone_picker(label: str, default_borough: str, default_zone: str, key: str) -> int:
    c1, c2 = st.columns([1, 2])
    borough = c1.selectbox(f"{label} borough", BOROUGHS, index=BOROUGHS.index(default_borough), key=f"{key}_b")
    options = known[known["Borough"] == borough].sort_values("Zone")
    names = options["Zone"].tolist()
    idx = names.index(default_zone) if default_zone in names else 0
    zone = c2.selectbox(f"{label} zone", names, index=idx, key=f"{key}_z")
    return int(options.index[options["Zone"] == zone][0])


st.title("🚕 NYC taxi trip duration")
st.caption("Estimate from information known at pickup only: zones, date and time. Trained on April 2026 yellow-cab trips.")
if quick:
    st.warning("Running on the small development models (`--quick`). Numbers are rough; train the full models for real estimates.")

pu = zone_picker("From", "Manhattan", "Midtown Center", "pu")
do = zone_picker("To", "Queens", "JFK Airport", "do")
c1, c2, c3 = st.columns(3)
day = c1.date_input("Date", pd.Timestamp("2026-06-10"))
at = c2.time_input("Pickup time", pd.Timestamp("2026-06-10 08:00").time(), step=900)
passengers = c3.number_input("Passengers", 1, 6, 1)

when = pd.Timestamp.combine(day, at)
point, lo, hi, dist = predict(pu, do, [when], passengers)

st.divider()
m1, m2, m3 = st.columns(3)
m1.metric("Estimated duration", f"{point[0]:.0f} min")
m2.metric("80% range", f"{lo[0]:.0f}–{hi[0]:.0f} min")
m3.metric("Straight-line distance", "same zone" if pu == do else f"{dist[0]:.1f} mi")
if not quick:  # the June numbers describe the full models only
    st.caption("8 out of 10 similar trips should fall inside the range. On June 2026 (not seen in training) the range covered "
               f"{bundle['meta']['june_coverage']:.1%} of trips; the estimate was off by {bundle['meta']['june_mae_min']:.1f} min on average.")

# ---- the same route across the chosen day --------------------------------------------------------------
times = pd.date_range(pd.Timestamp(day), periods=96, freq="15min")
p_day, lo_day, hi_day, _ = predict(pu, do, times, passengers)
curve = pd.DataFrame({"time": times, "estimate": p_day, "low": lo_day, "high": hi_day})
band = alt.Chart(curve).mark_area(opacity=0.25).encode(
    x=alt.X("time:T", title=None, axis=alt.Axis(format="%H:%M")),
    y=alt.Y("low:Q", title="minutes"), y2="high:Q")
line = alt.Chart(curve).mark_line().encode(x="time:T", y="estimate:Q",
                                          tooltip=[alt.Tooltip("time:T", format="%H:%M"),
                                                   alt.Tooltip("estimate:Q", format=".0f"),
                                                   alt.Tooltip("low:Q", format=".0f"), alt.Tooltip("high:Q", format=".0f")])
rule = alt.Chart(pd.DataFrame({"time": [when]})).mark_rule(strokeDash=[4, 4]).encode(x="time:T")
st.subheader(f"Across {pd.Timestamp(day):%A, %b %d}")
st.altair_chart((band + line + rule).properties(height=280), use_container_width=True)

daytime = curve[(curve["time"].dt.hour >= 6) & (curve["time"].dt.hour < 22)]
fast, slow = daytime.loc[daytime["estimate"].idxmin()], daytime.loc[daytime["estimate"].idxmax()]
st.write(f"Between 06:00 and 22:00 the quickest pickup is around **{fast['time']:%H:%M}** "
         f"(~{fast['estimate']:.0f} min) and the slowest around **{slow['time']:%H:%M}** (~{slow['estimate']:.0f} min).")

with st.expander("How this works and what it cannot do"):
    meta = bundle["meta"]
    st.markdown(f"""
- **Model:** LightGBM on zone ids, borough, airport flags, zone-centroid geometry and time of day/week
  (`{meta['point_model']}`), trained on {meta['n_train']:,} trips from {meta['train_month']}.
- **Range:** two quantile models (10th and 90th percentile), calibrated with conformalized quantile regression so that
  about 80% of trips fall inside.
- **Never used:** actual trip distance, fare, tips or dropoff time; none of them is known when the ride starts.
- **Limits:** no live traffic, weather or events; holidays and seasonal changes are not learned from one month;
  long trips (> 8 mi) are covered less often (about 74% on June). Vendor is fixed to the most common one.
- Full methodology and honest evaluation: [github.com/Venius24/ML-DATA](https://github.com/Venius24/ML-DATA).
""")
