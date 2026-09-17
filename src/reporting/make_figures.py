"""
Generate the figures used in the README from real pipeline outputs.

Why a separate script: the README should show real numbers, not hand-made
screenshots, so every figure it embeds must be reproducible by re-running the
pipeline (steps 1-7) then this script. Two figures (YOLO training curves,
QGIS map screenshot) cannot be generated headlessly -- they come straight
from the Colab run and from QGIS -- and are added to figures/ manually.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

FIGURES_DIR = Path("figures")

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.titleweight": "bold"})


def detection_class_distribution():
    """How many detections per distress class, from the real inference run."""
    df = pd.read_csv("outputs/detections/detections.csv")
    counts = df["class_name"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(counts.index, counts.values, color="#2b6cb0")
    ax.set_xlabel("Number of detections")
    ax.set_title("Detected distress instances by class (test-set inference)")
    for i, v in enumerate(counts.values):
        ax.text(v + 2, i, str(v), va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "detection_class_distribution.png")
    plt.close(fig)


def segment_condition_predicted_vs_actual():
    """Model prediction vs the real detection-based score, on the 8 drone-surveyed
    segments -- the honest check of the 'estimate condition with no imagery' claim."""
    df = pd.read_csv("outputs/segment_model/surveyed_vs_predicted.csv").sort_values("seg_id")

    x = range(len(df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - width / 2 for i in x], df["actual_score"], width, label="Actual (from detections)", color="#c53030")
    ax.bar([i + width / 2 for i in x], df["predicted_score"], width, label="Predicted (tabular model)", color="#2b6cb0")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"seg {i}" for i in df["seg_id"]])
    ax.set_ylabel("Priority score")
    ax.set_title("Segment condition model: predicted vs. actual (surveyed segments)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "segment_condition_predicted_vs_actual.png")
    plt.close(fig)


def copy_feature_importance():
    src = Path("outputs/segment_model/feature_importance.png")
    if src.exists():
        shutil.copy(src, FIGURES_DIR / "feature_importance.png")


def yolo_training_curves():
    """Redraw the Colab training log (models/baseline/results.csv, produced by
    ultralytics) in this project's own figure style, so it matches the other
    plots instead of looking like a pasted screenshot."""
    src = Path("models/baseline/results.csv")
    if not src.exists():
        return
    df = pd.read_csv(src)
    df.columns = [c.strip() for c in df.columns]

    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(df["epoch"], df["train/box_loss"], label="train box loss", color="#2b6cb0")
    ax_loss.plot(df["epoch"], df["val/box_loss"], label="val box loss", color="#2b6cb0", linestyle="--")
    ax_loss.plot(df["epoch"], df["train/cls_loss"], label="train cls loss", color="#c53030")
    ax_loss.plot(df["epoch"], df["val/cls_loss"], label="val cls loss", color="#c53030", linestyle="--")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training / validation loss")
    ax_loss.legend(fontsize=8)

    ax_metric.plot(df["epoch"], df["metrics/precision(B)"], label="precision", color="#2b6cb0")
    ax_metric.plot(df["epoch"], df["metrics/recall(B)"], label="recall", color="#c53030")
    ax_metric.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50", color="#38a169")
    ax_metric.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95", color="#805ad5")
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel("Score")
    ax_metric.set_title("Validation detection metrics")
    ax_metric.legend(fontsize=8)

    fig.suptitle("YOLO11n baseline training (50 epochs, imgsz 640, batch 16)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "yolo_training_curves.png")
    plt.close(fig)


def yolo_confusion_matrix():
    """Redraw the confusion matrix from the Colab validation run (baseline
    YOLO11n, 50 epochs) transcribed from ultralytics' own confusion_matrix.png,
    so it renders in this project's figure style.

    Note on the large "background" column: ultralytics computes this matrix
    across all candidate boxes at a low confidence threshold (to trace the
    full PR curve), so predicted-vs-background counts are not directly
    comparable to the operational precision/recall reported at conf=0.25 in
    the Results table -- read this matrix for *which classes get confused
    with each other*, not as a literal false-positive count.
    """
    classes = [
        "longitudinal_crack", "transverse_crack", "oblique_crack",
        "alligator_crack", "pothole", "repair", "background",
    ]
    # rows = predicted, columns = true (as labeled in the source plot)
    matrix = np.array([
        [259,  55,  11,  9,  2,  3, 13752],
        [ 13, 377,  22,  9,  0,  6, 19017],
        [ 27,  72,  88, 10,  1,  2,  9249],
        [ 10,  24,  13, 51,  1,  0,  3846],
        [  2,   2,   1,  1,  1,  0,  1667],
        [  3,  13,   1,  0,  0, 17,  1158],
        [ 25,  25,  24,  2,  2,  4,     0],
    ])
    # Log scale for color only -- raw counts span 4 orders of magnitude.
    display = np.log1p(matrix)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(display, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title("YOLO11n baseline: confusion matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if matrix[i, j] > 0:
                color = "white" if display[i, j] > display.max() * 0.6 else "black"
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, label="log(1 + count)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "yolo_confusion_matrix.png")
    plt.close(fig)


def qgis_priority_map():
    """A publication-quality static map: segments colored by priority score,
    the DBSCAN hotspot corridor, and a real basemap for geographic context.

    Why generate this instead of pasting a QGIS screenshot: a plain
    "right-click, apply graduated symbology" screenshot inherits QGIS's
    default styling (thin lines, default legend placement, no title/scale
    bar) and looks like a debug view, not a deliverable. This reproduces the
    same information -- it reads the exact GeoPackage QGIS would open --
    with deliberate cartographic styling (line weight, legend, scale bar,
    north arrow) so the README shows a finished map, and QGIS is still where
    you'd interactively explore the same file.
    """
    gpkg = Path("outputs/segment_priority.gpkg")
    if not gpkg.exists():
        return
    with open("configs/config.yaml", encoding="utf-8") as f:
        metric_crs = yaml.safe_load(f)["geo"]["metric_crs"]

    segs = gpd.read_file(gpkg, layer="segment_priority").to_crs(metric_crs)
    try:
        hotspots = gpd.read_file(gpkg, layer="distress_hotspots").to_crs(metric_crs)
    except Exception:
        hotspots = gpd.GeoDataFrame()

    fig, ax = plt.subplots(figsize=(8, 9))

    segs.plot(
        ax=ax, column="priority_score", cmap="RdYlGn_r", linewidth=5,
        legend=True, legend_kwds={"label": "Priority score (higher = repair sooner)", "shrink": 0.55},
        zorder=2,
    )
    if len(hotspots):
        # Plotted after (on top of) the segments, semi-transparent, so it
        # reads as a highlighted zone rather than being hidden underneath.
        hotspots.plot(ax=ax, color="#dd6b20", alpha=0.45, zorder=3)

    minx, miny, maxx, maxy = segs.total_bounds
    pad = max(maxx - minx, maxy - miny) * 0.3
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)

    import contextily as ctx
    ctx.add_basemap(ax, crs=metric_crs, source=ctx.providers.Esri.WorldGrayCanvas, zorder=1)

    # Axes are in a metric CRS (UTM), so a size bar in data units is exact metres.
    scalebar = AnchoredSizeBar(
        ax.transData, 300, "300 m", loc="lower right",
        pad=0.6, color="black", frameon=True, size_vertical=(maxy - miny) * 0.006,
        fontproperties={"size": 9},
    )
    ax.add_artist(scalebar)
    ax.annotate(
        "N", xy=(0.05, 0.90), xytext=(0.05, 0.78), xycoords="axes fraction",
        arrowprops=dict(facecolor="black", width=3, headwidth=10),
        ha="center", fontsize=11, fontweight="bold",
    )

    ax.set_axis_off()
    ax.set_title(
        "Pavement maintenance priority — Chemin de la Côte-Sainte-Catherine, Montréal\n"
        "(orange band = DBSCAN distress hotspot; synthetic demo, see README)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "qgis_priority_map.png", dpi=200)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    detection_class_distribution()
    segment_condition_predicted_vs_actual()
    copy_feature_importance()
    yolo_training_curves()
    yolo_confusion_matrix()
    qgis_priority_map()
    print(f"Figures written to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
