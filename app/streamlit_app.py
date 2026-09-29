"""
Interactive demo of the pavement-distress pipeline (Streamlit).

    streamlit run app/streamlit_app.py

How Streamlit works, in one sentence: this is a normal top-to-bottom Python
script, every st.something() call draws one element on the page, and the whole
script re-runs each time the user interacts (uploads a file, moves a slider).
That's why the expensive parts (loading the model, reading files) are wrapped
in @st.cache_* decorators: they run once and are reused on every re-run.

Three tabs:
  1. Detect distress -- run the trained YOLO model on an uploaded or sample image
  2. Priority map    -- the 32 scored street segments + DBSCAN hotspots, interactive
  3. Condition model -- how the scikit-learn PCI model performs vs. the city's real data
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import pandas as pd
import pydeck as pdk
import streamlit as st
from pydeck.types import String
import yaml
from PIL import Image
from ultralytics import YOLO

# Paths are resolved from this file's location (not the current directory), so
# the app works whether it's launched from the repo root or from app/.
ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/akrambouznad789-cyber/pavement-distress-drone"

st.set_page_config(page_title="Pavement distress from drone imagery", page_icon="🛣️", layout="wide")


# ---- Cached loaders: run once, reused on every re-run -----------------------
@st.cache_data
def load_config() -> dict:
    with open(ROOT / "configs/config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource  # cache_resource = for objects like models (not copied on each use)
def load_model() -> YOLO:
    return YOLO(str(ROOT / load_config()["inference"]["weights"]))


@st.cache_data
def load_map_layers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    gpkg = ROOT / "outputs/segment_priority.gpkg"
    segments = gpd.read_file(gpkg, layer="segment_priority")
    hotspots = gpd.read_file(gpkg, layer="distress_hotspots")
    # Add the city's real PCI and the model's predicted PCI to each segment (NaN = no match).
    pci = pd.read_csv(ROOT / "outputs/segment_model/surveyed_vs_predicted.csv")
    segments = segments.merge(pci[["seg_id", "real_indice_pci", "predicted_score"]], on="seg_id", how="left")
    return segments, hotspots


cfg = load_config()
CLASSES = cfg["data"]["classes"]
SEVERITY = cfg["geo"]["severity_weights"]

# ---- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.markdown(
        "Drone images → **YOLO** distress detection → georeferencing → **DBSCAN** hotspots → "
        "per-street **priority score** → **scikit-learn** condition model trained on ~16,000 "
        "real City of Montréal segments."
    )
    st.markdown(
        "The images, detector, street network and condition data are real. "
        "The GPS position of each survey photo is **simulated** (no real flight was available)."
    )
    st.markdown(f"[Source code & full write-up]({REPO_URL})")

st.title("🛣️ Pavement distress detection & maintenance priority")

tab_detect, tab_map, tab_model = st.tabs(["1 · Detect distress", "2 · Priority map", "3 · Condition model"])

# ---- Tab 1: detection ----------------------------------------------------------
with tab_detect:
    st.markdown(
        "Upload a **top-down drone photo of a road**, or pick a sample. The model was trained on "
        "UAV-PDD2023 (6 classes: longitudinal, transverse, oblique and alligator cracks, potholes, "
        "repairs), so it works best on similar nadir imagery."
    )
    col_input, col_conf = st.columns([3, 1])
    with col_conf:
        conf = st.slider("Confidence threshold", 0.05, 0.90, float(cfg["inference"]["conf"]), 0.05,
                         help="Boxes below this confidence are hidden. Lower = more detections, more false alarms.")
    with col_input:
        uploaded = st.file_uploader("Your image", type=["jpg", "jpeg", "png"])
        samples = sorted((ROOT / "assets/sample_images").glob("*.jpg"))
        sample = st.selectbox("…or a sample image", samples, format_func=lambda p: p.name,
                              disabled=uploaded is not None)

    image = Image.open(uploaded if uploaded is not None else sample).convert("RGB")
    result = load_model().predict(image, conf=conf, verbose=False)[0]

    col_img, col_table = st.columns([3, 2])
    with col_img:
        # result.plot() draws the boxes and returns a BGR array (OpenCV convention) -> flip to RGB.
        st.image(result.plot()[:, :, ::-1], caption=f"{len(result.boxes)} detections", width="stretch")
    with col_table:
        detections = pd.DataFrame(
            [{"class": CLASSES[int(b.cls)], "confidence": round(float(b.conf), 2)} for b in result.boxes]
        )
        if detections.empty:
            st.info("No distress detected at this confidence threshold.")
        else:
            # Same formula the pipeline uses per street segment: sum of severity x confidence.
            detections["severity"] = detections["class"].map(SEVERITY)
            score = (detections["severity"] * detections["confidence"]).sum()
            st.metric("Severity score of this image", f"{score:.1f}",
                      help="Σ severity(class) × confidence. Summed over all photos of a street, "
                           "this becomes the street's priority score.")
            st.dataframe(detections.sort_values("confidence", ascending=False),
                         hide_index=True, width="stretch")

# ---- Tab 2: priority map ---------------------------------------------------------
with tab_map:
    segments, hotspots = load_map_layers()
    st.markdown(
        f"**{len(segments)} real street segments** (OpenStreetMap) near Polytechnique Montréal, "
        "colored by maintenance priority. Hover a street for its scores; orange areas are DBSCAN "
        "hotspots (≥ 3 detections within 20 m)."
    )
    show_hotspots = st.checkbox("Show DBSCAN hotspots", value=True)

    # Color each segment from green (low priority) to red (high), like the README map.
    cmap = matplotlib.colormaps["RdYlGn_r"]
    max_score = segments["priority_score"].max()
    segments["color"] = segments["priority_score"].apply(
        lambda s: [int(255 * c) for c in cmap(s / max_score)[:3]]
    )
    # pydeck wants plain lists of [lon, lat] points, not shapely geometries.
    segments["path"] = segments.geometry.apply(lambda g: [list(xy) for xy in g.coords])
    # Tooltip values as text: "no match" where the segment has no real Montréal PCI, "none"
    # where the drone found nothing. Why it matters: pydeck sends the data to the browser as
    # JSON, and a missing value (NaN) is not valid JSON, so a single NaN anywhere makes the
    # whole map fail to render.
    as_text = lambda col: segments[col].round(1).map(lambda v: "no match" if pd.isna(v) else f"{v:.1f}")
    seg_data = segments.assign(
        priority=segments["priority_score"].round(1), real_pci=as_text("real_indice_pci"),
        pred_pci=as_text("predicted_score"), worst_class=segments["worst_class"].fillna("none"),
    ).drop(columns=["geometry", "real_indice_pci", "predicted_score"])
    # Safety net for any other missing value: None becomes JSON null, which is valid.
    seg_data = seg_data.astype(object).where(seg_data.notna(), None)

    # Careful: pydeck reads every plain string as a column name ("path" -> each row's path).
    # A literal text value like "pixels" must be wrapped in String(), otherwise pydeck looks
    # for a "pixels" column, the unit is lost, and the lines are drawn thousands of pixels wide.
    layers = [
        pdk.Layer("PathLayer", seg_data, get_path="path", get_color="color", get_width=6,
                  width_units=String("pixels"), cap_rounded=True, joint_rounded=True, pickable=True)
    ]
    if show_hotspots:
        hotspots["polygon"] = hotspots.geometry.apply(lambda g: [list(xy) for xy in g.exterior.coords])
        layers.append(pdk.Layer("PolygonLayer", hotspots.drop(columns="geometry"), get_polygon="polygon",
                                get_fill_color=[221, 107, 32, 90], get_line_color=[221, 107, 32, 160],
                                pickable=False))

    center = segments.union_all().centroid
    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=center.y, longitude=center.x, zoom=15),
        map_provider="carto", map_style="light",
        tooltip={"html": "<b>{name}</b> (segment {seg_id})<br/>"
                         "Priority score: <b>{priority}</b> (rank {priority_rank})<br/>"
                         "Detections: {n_detections} · most severe: {worst_class}<br/>"
                         "Real PCI (city): {real_pci} · predicted PCI: {pred_pci}"},
    ))

    st.subheader("Top 10 segments to repair first")
    st.dataframe(
        seg_data.sort_values("priority_rank")
        [["priority_rank", "seg_id", "name", "priority", "n_detections", "worst_class", "real_pci"]]
        .head(10)
        .rename(columns={"priority_rank": "rank", "worst_class": "most severe class",
                         "n_detections": "detections", "real_pci": "real PCI (city)"}),
        hide_index=True,
    )

# ---- Tab 3: condition model ------------------------------------------------------------
with tab_model:
    st.markdown(
        "A drone can't survey a whole city every year, so a second model predicts each street's "
        "**Pavement Condition Index** (PCI, 0-100, higher = better) from attributes the city "
        "already has everywhere: length, borough, roughness (IRI), pavement age and material. "
        "It's a gradient-boosting model trained on **~16,000 real segments** from the City of "
        "Montréal's pavement survey."
    )
    with open(ROOT / "outputs/segment_model/metrics.json", encoding="utf-8") as f:
        test = json.load(f)["test"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Test R²", f"{test['r2']:.2f}", help="Share of the variation in real PCI the model explains.")
    c2.metric("Mean absolute error", f"{test['mae']:.1f} PCI pts")
    c3.metric("RMSE", f"{test['rmse']:.1f} PCI pts")

    st.image(str(ROOT / "figures/segment_condition_predicted_vs_actual.png"), width="stretch")
    st.markdown(
        "**Honest reading:** the model tracks the real PCI moderately (rank correlation 0.48) but "
        "compresses its predictions toward the city average, overestimating the worst streets. The "
        "drone score shows no relation to the real PCI (−0.13), which is expected here: survey "
        "positions are simulated, so a detection's street carries no information. Validating it "
        "needs a real geotagged flight."
    )
    st.image(str(ROOT / "figures/feature_importance.png"), caption="What the model relies on", width=550)
