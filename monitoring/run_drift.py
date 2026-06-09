"""
Drift detection script using Evidently AI.

Produces two reports saved to the Evidently workspace:
  1. Prediction + confidence + image property drift  (explicit ColumnDriftMetric per column)
  2. Embedding drift — domain classifier method      (EmbeddingsDriftMetric)

Usage:
    poetry run python -m monitoring.run_drift

Then view the dashboard:
    poetry run evidently ui --workspace monitoring/workspace --port 8001
    -> http://localhost:8001

Port map:
    8000 -- FastAPI (inference API)
    8001 -- Evidently UI (monitoring dashboard)
    5000 -- MLflow UI
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    EmbeddingsDriftMetric,
)
from evidently.metrics.data_drift.embedding_drift_methods import model as classifier_method
from evidently.report import Report
from evidently.ui.dashboards import (
    CounterAgg,
    DashboardPanelCounter,
    DashboardPanelPlot,
    PanelValue,
    PlotType,
    ReportFilter,
)
from evidently.ui.workspace import Workspace
from loguru import logger

from garbage_classification.config import PROCESSED_DATA_DIR, PROJ_ROOT

PREDICTIONS_LOG = PROJ_ROOT / "logs" / "predictions.jsonl"
REFERENCE_PATH = PROJ_ROOT / "monitoring" / "reference.parquet"
WORKSPACE_DIR = PROJ_ROOT / "monitoring" / "workspace"
STATIC_DIR = PROJ_ROOT / "monitoring" / "static"
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
PROJECT_NAME = "WasteNet Garbage Classifier"

# Prefix used for embedding columns in both reference.parquet and predictions.jsonl
_EMB_PREFIX = "emb_"


def _get_emb_cols(df: pd.DataFrame) -> list[str]:
    """Return sorted embedding column names (emb_0, emb_1, ...) found in df."""
    return sorted(
        [c for c in df.columns if c.startswith(_EMB_PREFIX)],
        key=lambda c: int(c.split("_")[1]),
    )


def load_reference_data() -> pd.DataFrame:
    """Load pre-computed reference embeddings from monitoring/reference.parquet.

    Build it first with: poetry run python -m monitoring.build_reference
    """
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Reference file not found: {REFERENCE_PATH}\n"
            "Run first: poetry run python -m monitoring.build_reference"
        )
    return pd.read_parquet(REFERENCE_PATH)


def load_production_data() -> pd.DataFrame:
    """Load logged predictions from the API and flatten the embedding list column."""
    if not PREDICTIONS_LOG.exists():
        raise FileNotFoundError(
            f"No predictions log found at {PREDICTIONS_LOG}. "
            "Make some predictions via the API first."
        )
    records = [
        json.loads(line)
        for line in PREDICTIONS_LOG.read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("Predictions log is empty.")

    df = pd.DataFrame(records)

    # Flatten the embedding list stored as a JSON array into individual emb_N columns
    if "embedding" in df.columns:
        emb_matrix = np.array(df["embedding"].tolist(), dtype=np.float32)
        emb_df = pd.DataFrame(
            emb_matrix,
            columns=[f"{_EMB_PREFIX}{i}" for i in range(emb_matrix.shape[1])],
        )
        df = pd.concat([df.drop(columns=["embedding"]), emb_df], axis=1)

    return df


def _setup_dashboard_panels(project) -> None:
    """Configure monitoring dashboard panels — 3 logical rows.

    Row 1 — Counters  : quick status snapshot (last run values)
    Row 2 — Global    : overall drift signals over time
    Row 3 — Breakdown : per-signal detail, grouped by topic
    """
    no_filter = ReportFilter(metadata_values={}, tag_values=[])

    # ── Row 1: Counters ───────────────────────────────────────────────────────
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Share of drifted features (last run)",
            filter=no_filter,
            value=PanelValue(
                metric_id="DatasetDriftMetric",
                field_path="share_of_drifted_columns",
                legend="share",
            ),
            text="",
            agg=CounterAgg.LAST,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Embedding drift score / ROC-AUC (last run)",
            filter=no_filter,
            value=PanelValue(
                metric_id="EmbeddingsDriftMetric",
                metric_args={"embeddings_name": "cnn_embedding"},
                field_path="drift_score",
                legend="ROC-AUC",
            ),
            text="",
            agg=CounterAgg.LAST,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Missing values share (last run)",
            filter=no_filter,
            value=PanelValue(
                metric_id="DatasetMissingValuesMetric",
                field_path="current.share_of_missing_values",
                legend="missing",
            ),
            text="",
            agg=CounterAgg.LAST,
            size=1,
        )
    )

    # ── Row 2: Global drift over time ─────────────────────────────────────────
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Drifted features share over time",
            filter=no_filter,
            values=[
                PanelValue(
                    metric_id="DatasetDriftMetric",
                    field_path="share_of_drifted_columns",
                    legend="share of drifted features",
                ),
            ],
            plot_type=PlotType.LINE,
            size=2,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Embedding drift score over time (ROC-AUC)",
            filter=no_filter,
            values=[
                PanelValue(
                    metric_id="EmbeddingsDriftMetric",
                    metric_args={"embeddings_name": "cnn_embedding"},
                    field_path="drift_score",
                    legend="ROC-AUC",
                ),
            ],
            plot_type=PlotType.LINE,
            size=2,
        )
    )

    # ── Row 3a: Prediction signals ────────────────────────────────────────────
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Predicted class drift score over time",
            filter=no_filter,
            values=[
                PanelValue(
                    metric_id="ColumnDriftMetric",
                    metric_args={"column_name.name": "predicted_class"},
                    field_path="drift_score",
                    legend="predicted_class",
                ),
            ],
            plot_type=PlotType.BAR,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Confidence drift score over time",
            filter=no_filter,
            values=[
                PanelValue(
                    metric_id="ColumnDriftMetric",
                    metric_args={"column_name.name": "confidence"},
                    field_path="drift_score",
                    legend="confidence",
                ),
            ],
            plot_type=PlotType.LINE,
            size=1,
        )
    )
    # ── Row 3b: Image quality signals (grouped) ───────────────────────────────
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Image quality drift (brightness & sharpness)",
            filter=no_filter,
            values=[
                PanelValue(
                    metric_id="ColumnDriftMetric",
                    metric_args={"column_name.name": "brightness"},
                    field_path="drift_score",
                    legend="brightness",
                ),
                PanelValue(
                    metric_id="ColumnDriftMetric",
                    metric_args={"column_name.name": "blur_score"},
                    field_path="drift_score",
                    legend="blur_score",
                ),
            ],
            plot_type=PlotType.LINE,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Color channel drift (R / G / B)",
            filter=no_filter,
            values=[
                PanelValue(
                    metric_id="ColumnDriftMetric",
                    metric_args={"column_name.name": "r_mean"},
                    field_path="drift_score",
                    legend="R",
                ),
                PanelValue(
                    metric_id="ColumnDriftMetric",
                    metric_args={"column_name.name": "g_mean"},
                    field_path="drift_score",
                    legend="G",
                ),
                PanelValue(
                    metric_id="ColumnDriftMetric",
                    metric_args={"column_name.name": "b_mean"},
                    field_path="drift_score",
                    legend="B",
                ),
            ],
            plot_type=PlotType.LINE,
            size=1,
        )
    )

    project.save()
    logger.info("Dashboard panels configured.")


def _get_or_create_project(ws: Workspace):
    projects = ws.list_projects()
    project = next((p for p in projects if p.name == PROJECT_NAME), None)
    if project is None:
        project = ws.create_project(PROJECT_NAME)
        project.description = "Drift monitoring for the WasteNet garbage classifier"
        project.save()
        _setup_dashboard_panels(project)
    return project


def run() -> None:
    logger.info("Loading reference data (training set)...")
    ref_df = load_reference_data()

    logger.info(f"Loading production data from {PREDICTIONS_LOG}...")
    prod_df = load_production_data()
    logger.info(f"Production samples: {len(prod_df)}")

    ws = Workspace.create(str(WORKSPACE_DIR))
    project = _get_or_create_project(ws)

    # -------------------------------------------------------------------------
    # Report 1: Prediction drift + confidence + image property drift
    # Columns tested:
    #   predicted_class -> chi-squared (categorical)
    #   confidence      -> K-S test    (numerical)
    #   brightness      -> K-S test    (numerical)
    #   blur_score      -> K-S test    (numerical)
    #   r/g/b_mean      -> K-S test    (numerical)
    # -------------------------------------------------------------------------
    scalar_cols = [
        "predicted_class", "confidence",
        "brightness", "blur_score",
        "r_mean", "g_mean", "b_mean",
    ]
    # Use only columns available in both dataframes
    scalar_cols = [c for c in scalar_cols if c in ref_df.columns and c in prod_df.columns]

    report1_cols = scalar_cols
    logger.info("Running Report 1: prediction + confidence + image property drift...")
    report1 = Report(
        metrics=[
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            *[ColumnDriftMetric(column_name=c) for c in report1_cols],
        ],
        tags=["data-drift", "scalar-features"],
    )
    report1.run(
        reference_data=ref_df[report1_cols],
        current_data=prod_df[report1_cols],
        column_mapping=None,
    )
    ws.add_report(project.id, report1)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    report1.save_html(str(STATIC_DIR / "index.html"))
    logger.success(f"Report 1 saved (workspace + HTML -> {STATIC_DIR / 'index.html'}).")

    # -------------------------------------------------------------------------
    # Report 2: Embedding drift via domain classifier (ROC-AUC method)
    # A binary classifier is trained to separate reference vs production embeddings.
    # Drift is detected when ROC-AUC > 0.55 (threshold configurable).
    # Skipped when fewer than 20 production samples are available (not enough to train
    # the internal classifier reliably).
    # -------------------------------------------------------------------------
    emb_cols = _get_emb_cols(ref_df)
    prod_emb_cols = _get_emb_cols(prod_df)
    common_emb_cols = [c for c in emb_cols if c in prod_emb_cols]

    if common_emb_cols and len(prod_df) >= 20:
        col_mapping = ColumnMapping(embeddings={"cnn_embedding": common_emb_cols})
        logger.info(
            f"Running Report 2: embedding drift "
            f"({len(common_emb_cols)} dims, domain classifier, threshold=0.55)..."
        )
        report2 = Report(
            metrics=[
                EmbeddingsDriftMetric(
                    embeddings_name="cnn_embedding",
                    drift_method=classifier_method(threshold=0.55),
                )
            ],
            tags=["embedding-drift", "cnn-embedding"],
        )
        report2.run(
            reference_data=ref_df[common_emb_cols],
            current_data=prod_df[common_emb_cols],
            column_mapping=col_mapping,
        )
        ws.add_report(project.id, report2)
        report2.save_html(str(STATIC_DIR / "embedding_drift.html"))
        logger.success(
            f"Report 2 (embedding drift) saved (workspace + HTML -> "
            f"{STATIC_DIR / 'embedding_drift.html'})."
        )
    elif not common_emb_cols:
        logger.warning(
            "No embedding columns found. "
            "Re-run build_reference.py and restart the API to enable embedding drift."
        )
    else:
        logger.warning(
            f"Skipping embedding drift: need >= 20 production samples (got {len(prod_df)}). "
            "Make more predictions via the API and re-run."
        )

    logger.info(f"Launch dashboard: evidently ui --workspace {WORKSPACE_DIR} --port 8001")
    logger.info("Then open: http://localhost:8001")


if __name__ == "__main__":
    run()
