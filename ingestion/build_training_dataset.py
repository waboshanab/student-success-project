"""ingestion.build_training_dataset

Build a term-level training dataset from OULAD raw CSVs.

Design notes / assumptions (documented inline):
- Expects raw CSVs to live under `data/raw/oulad/anonymisedData/` with the following files:
  - studentRegistration.csv: `id_student`, `code_module`, `code_presentation`, `date_registration`, `date_unregistration`
  - studentInfo.csv: `id_student`, `code_module`, `code_presentation`, `final_result`, plus demographic fields
  - assessments.csv: `id_assessment`, `code_module`, `code_presentation`, `date`, `weight`, `assessment_type`
  - studentAssessment.csv: `id_assessment`, `id_student`, `date_submitted`, `score`, `is_banked` (no module/presentation; joined via assessments)
  - studentVle.csv: `id_student`, `code_module`, `code_presentation`, `id_site`, `date`, `sum_click`
  - vle.csv: metadata for VLE activities
- Column name normalization: `id_student` → `student_id` during load
- Date fields are numeric offsets (days/weeks from start); coerced to numeric type
- Time-awareness: events after unregistration date are excluded from feature aggregation
- Missing values: counts filled with 0; means left as NaN for downstream modeling decisions

Run as script or import functions in tests.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd


# Configure module-level logger
logger = logging.getLogger("ingestion.build_training_dataset")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def load_raw_data(raw_dir: str) -> Dict[str, pd.DataFrame]:
    """Load raw OULAD CSV files into pandas DataFrames."""
    logger.info("Loading raw CSVs from %s", raw_dir)

    def _read_csv(fname: str, **kwargs) -> Optional[pd.DataFrame]:
        path = os.path.join(raw_dir, fname)
        if not os.path.exists(path):
            logger.warning("File not found: %s", path)
            return None
        logger.debug("Reading %s", path)
        return pd.read_csv(path, **kwargs)

    dfs: Dict[str, Optional[pd.DataFrame]] = {
        "courses": _read_csv("courses.csv"),
        "studentInfo": _read_csv("studentInfo.csv"),
        "studentRegistration": _read_csv("studentRegistration.csv"),
        "assessments": _read_csv("assessments.csv"),
        "studentAssessment": _read_csv("studentAssessment.csv"),
        "studentVle": _read_csv("studentVle.csv"),
        "vle": _read_csv("vle.csv"),
    }

    # Convert Optional -> raise helpful errors if core tables missing
    missing = [k for k, v in dfs.items() if v is None and k in ("studentRegistration", "studentInfo")]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")

    # Normalize common columns and coerce known numeric date-like fields (OULAD uses offsets)
    for name, df in list(dfs.items()):
        if df is None:
            continue
        # normalize id column
        if "id_student" in df.columns and "student_id" not in df.columns:
            df.rename(columns={"id_student": "student_id"}, inplace=True)
        # coerce numeric date-like fields to numeric (prefer numeric offsets)
        for col in ("date_registration", "date_unregistration", "date_submitted", "date"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        dfs[name] = df

    # Return only existing DataFrames
    return {k: v for k, v in dfs.items() if v is not None}


def build_enrollment_base(student_registration: pd.DataFrame, student_info: pd.DataFrame) -> pd.DataFrame:
    """Create a student-term enrollment base keyed by (student_id, code_module, code_presentation).

    The base includes registration/unregistration dates and final_result from studentInfo.
    """
    logger.info("Building enrollment base")
    reg = student_registration.copy()

    # Normalize column names if necessary
    if "id_student" not in reg.columns and "student_id" in reg.columns:
        reg = reg.rename(columns={"student_id": "id_student"})

    # Keep only necessary fields
    reg_cols = [c for c in ["id_student", "code_module", "code_presentation", "date_registration", "date_unregistration"] if c in reg.columns]
    reg = reg[reg_cols]

    # Ensure id column name is consistent
    reg = reg.rename(columns={"id_student": "student_id"})

    info = student_info.copy()
    if "id_student" in info.columns:
        info = info.rename(columns={"id_student": "student_id"})

    # select relevant columns from info: include code_module, code_presentation (for merge key), final_result and demographics
    keep_info = [c for c in ["student_id", "code_module", "code_presentation", "final_result", "age_band", "gender", "disability"] if c in info.columns]
    info = info[keep_info]

    base = pd.merge(
        reg,
        info,
        how="left",
        on=["student_id", "code_module", "code_presentation"],
    )

    # Create risk target
    base["risk_flag"] = base["final_result"].fillna("").astype(str).str.strip().str.lower().apply(lambda x: 1 if x == "withdrawn" else 0)

    logger.debug("Enrollment base rows: %d", len(base))
    return base


def _filter_by_unregistration(events: pd.DataFrame, base: pd.DataFrame, event_date_col: str) -> pd.DataFrame:
    """Filter events occurring after unregistration using numeric-first logic.

    Strategy:
      1) Merge events with enrollment base on (student_id, code_module, code_presentation) to obtain `date_unregistration`.
      2) Try numeric comparison first (OULAD often uses numeric offsets). If numeric data exists, use it.
      3) Fallback to datetime comparison if numeric not available.
      4) If neither available, keep all events and log a warning.
    """
    df = events.copy()
    if "id_student" in df.columns and "student_id" not in df.columns:
        df = df.rename(columns={"id_student": "student_id"})

    merged = pd.merge(
        df,
        base[["student_id", "code_module", "code_presentation", "date_unregistration"]],
        how="left",
        on=["student_id", "code_module", "code_presentation"],
    )

    if "date_unregistration" not in merged.columns:
        logger.debug("No unregistration dates available; skipping time-based filtering")
        return df

    # Try numeric comparison first
    ev_num = pd.to_numeric(merged[event_date_col], errors="coerce")
    un_num = pd.to_numeric(merged["date_unregistration"], errors="coerce")
    numeric_mask = ev_num.notna() & un_num.notna()
    if numeric_mask.any():
        mask = merged["date_unregistration"].isna() | (ev_num <= un_num)
        filtered = merged[mask]
        logger.debug("Filtered events (numeric): %d -> %d", len(df), len(filtered))
        return filtered.drop(columns=["date_unregistration"])

    # Fallback to datetime comparison
    ev_dt = pd.to_datetime(merged[event_date_col], errors="coerce")
    un_dt = pd.to_datetime(merged["date_unregistration"], errors="coerce")
    if ev_dt.notna().any() and un_dt.notna().any():
        mask = merged["date_unregistration"].isna() | (ev_dt <= un_dt)
        filtered = merged[mask]
        logger.debug("Filtered events (datetime): %d -> %d", len(df), len(filtered))
        return filtered.drop(columns=["date_unregistration"])

    logger.warning("Could not apply unregistration-based filtering due to incompatible date types; keeping all events")
    return df


def aggregate_lms_features(student_vle: pd.DataFrame, enrollment_base: pd.DataFrame) -> pd.DataFrame:
    """Aggregate LMS engagement into student-term level features.

    Generated features:
      - total_lms_clicks
      - active_lms_days
      - mean_clicks_per_day

    Assumptions: `studentVle` has a clicks column named one of: `sum_click`, `clicks`, `num_clicks`.
    Date column may be named `date` and can be datetime or numeric (days/weeks since start).
    """
    logger.info("Aggregating LMS features")
    df = student_vle.copy()

    # Normalize student id
    if "id_student" in df.columns and "student_id" not in df.columns:
        df = df.rename(columns={"id_student": "student_id"})

    # Identify clicks column
    possible_click_cols = [c for c in ["sum_click", "clicks", "num_clicks", "activity"] if c in df.columns]
    if not possible_click_cols:
        logger.warning("No clicks column found in studentVle; creating a synthetic clicks=1 per record")
        df["_clicks"] = 1
        clicks_col = "_clicks"
    else:
        clicks_col = possible_click_cols[0]

    # Ensure event filtering by unregistration date
    df_filtered = _filter_by_unregistration(df, enrollment_base, event_date_col="date")

    # Compute per-day granularity for active days: if date is datetime, use date only
    if pd.api.types.is_datetime64_any_dtype(df_filtered["date"]):
        df_filtered["_activity_day"] = df_filtered["date"].dt.floor("D")
    else:
        # If numeric, treat each unique numeric value as a day/week
        df_filtered["_activity_day"] = df_filtered["date"]

    group_cols = ["student_id", "code_module", "code_presentation"]
    agg = (
        df_filtered.groupby(group_cols)
        .agg(
            total_lms_clicks=(clicks_col, "sum"),
            active_lms_days=("_activity_day", "nunique"),
        )
        .reset_index()
    )

    # mean clicks per active day (vectorized)
    agg["mean_clicks_per_day"] = (agg["total_lms_clicks"] / agg["active_lms_days"].replace({0: pd.NA})).fillna(0).astype(float)

    # Ensure types
    agg["total_lms_clicks"] = agg["total_lms_clicks"].fillna(0).astype(float)
    agg["active_lms_days"] = agg["active_lms_days"].fillna(0).astype(int)

    logger.debug("LMS feature rows: %d", len(agg))
    return agg


def aggregate_assessment_features(student_assessment: pd.DataFrame, assessments: pd.DataFrame, enrollment_base: pd.DataFrame) -> pd.DataFrame:
    """Aggregate assessment features per student-term.

    Features:
      - num_assessments_submitted
      - mean_assessment_score

    Assumptions: submissions are indicated by non-null `date_submitted` or non-null `score`.
    Requires assessments table to join code_module and code_presentation.
    """
    logger.info("Aggregating assessment features")
    df = student_assessment.copy()

    # Normalize id
    if "id_student" in df.columns and "student_id" not in df.columns:
        df = df.rename(columns={"id_student": "student_id"})

    # Join with assessments to get code_module and code_presentation
    if not assessments.empty and "id_assessment" in assessments.columns:
        assess_cols = [c for c in ["id_assessment", "code_module", "code_presentation"] if c in assessments.columns]
        df = df.merge(
            assessments[assess_cols],
            how="left",
            on="id_assessment",
        )
        logger.debug("Merged with assessments table; rows: %d", len(df))

    # Filter events after unregistration
    # use date_submitted if available; otherwise fall back to submission flags
    date_col = "date_submitted" if "date_submitted" in df.columns else None

    if date_col is not None and "code_module" in df.columns and "code_presentation" in df.columns:
        df_filtered = _filter_by_unregistration(df, enrollment_base, event_date_col=date_col)
    else:
        logger.debug("Skipping unregistration-based filtering (missing date or module/presentation columns)")
        df_filtered = df

    # Define submitted rows
    submitted_mask = pd.Series(False, index=df_filtered.index)
    if date_col is not None:
        submitted_mask = df_filtered[date_col].notna()
    elif "score" in df_filtered.columns:
        submitted_mask = df_filtered["score"].notna()

    df_submitted = df_filtered[submitted_mask]

    group_cols = ["student_id", "code_module", "code_presentation"]
    # Check that all group cols exist
    group_cols = [c for c in group_cols if c in df_submitted.columns]
    
    if len(df_submitted) == 0 or len(group_cols) == 0:
        logger.warning("No submitted assessments or missing grouping columns; returning empty assessment features")
        return pd.DataFrame(columns=["student_id", "code_module", "code_presentation", "num_assessments_submitted", "mean_assessment_score"])

    agg = (
        df_submitted.groupby(group_cols)
        .agg(
            num_assessments_submitted=("score", "count") if "score" in df_submitted.columns else ("id_assessment", "count"),
            mean_assessment_score=("score", "mean") if "score" in df_submitted.columns else ("id_assessment", lambda s: np.nan),
        )
        .reset_index()
    )

    # Explicitly cast
    agg["num_assessments_submitted"] = agg["num_assessments_submitted"].fillna(0).astype(int)

    logger.debug("Assessment feature rows: %d", len(agg))
    return agg


def assemble_training_dataset(
    enrollment_base: pd.DataFrame,
    lms_features: pd.DataFrame,
    assessment_features: pd.DataFrame,
) -> pd.DataFrame:
    """Merge enrollment, LMS and assessment features into the final training table.

    Missing value handling decisions (explicit):
      - Counts (total_lms_clicks, active_lms_days, num_assessments_submitted) -> fill 0
      - mean_clicks_per_day -> fill 0 when active_lms_days == 0
      - mean_assessment_score -> leave as NaN (modeler should decide on imputation); we document this.
    """
    logger.info("Assembling final training dataset")

    df = enrollment_base.copy()

    # Merge LMS
    df = df.merge(
        lms_features,
        on=["student_id", "code_module", "code_presentation"],
        how="left",
    )

    # Merge assessment
    df = df.merge(
        assessment_features,
        on=["student_id", "code_module", "code_presentation"],
        how="left",
    )

    # Fill counts
    df["total_lms_clicks"] = df["total_lms_clicks"].fillna(0)
    df["active_lms_days"] = df["active_lms_days"].fillna(0).astype(int)
    df["num_assessments_submitted"] = df["num_assessments_submitted"].fillna(0).astype(int)

    # mean_clicks_per_day: if NaN (no active days) -> 0
    df["mean_clicks_per_day"] = df["mean_clicks_per_day"].fillna(0.0)

    # mean_assessment_score left as NaN to let downstream models decide on imputation strategy

    final_cols = [
        "student_id",
        "code_module",
        "code_presentation",
        "date_registration",
        "date_unregistration",
        "final_result",
        "risk_flag",
        "total_lms_clicks",
        "active_lms_days",
        "mean_clicks_per_day",
        "num_assessments_submitted",
        "mean_assessment_score",
    ]

    # Keep only available columns from final_cols
    keep = [c for c in final_cols if c in df.columns]
    result = df[keep].copy()

    logger.info("Final training dataset rows: %d", len(result))
    return result


def write_parquet(df: pd.DataFrame, out_path: str) -> None:
    logger.info("Writing training dataset to %s", out_path)
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info("Wrote %d rows to %s", len(df), out_path)


def main(raw_dir: str = "data/raw/oulad/anonymisedData", out_path: str = "data/processed/training_dataset.parquet") -> None:
    logger.info("Starting build_training_dataset pipeline")
    dfs = load_raw_data(raw_dir)

    enrollment_base = build_enrollment_base(dfs["studentRegistration"], dfs["studentInfo"])

    lms_feats = aggregate_lms_features(dfs.get("studentVle", pd.DataFrame()), enrollment_base)
    assess_feats = aggregate_assessment_features(dfs.get("studentAssessment", pd.DataFrame()), dfs.get("assessments", pd.DataFrame()), enrollment_base)

    training = assemble_training_dataset(enrollment_base, lms_feats, assess_feats)

    write_parquet(training, out_path)
    logger.info("Pipeline finished successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build term-level training dataset from OULAD raw files")
    parser.add_argument("--raw-dir", default="data/raw/oulad/anonymisedData", help="Path to OULAD raw folder")
    parser.add_argument(
        "--out", default="data/processed/training_dataset.parquet", help="Output parquet path"
    )
    args = parser.parse_args()
    main(raw_dir=args.raw_dir, out_path=args.out)
