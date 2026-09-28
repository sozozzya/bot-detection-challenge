from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

SEED = 42

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
EVENTS_PATH = DATA_DIR / "events.csv.gz"
SUBMISSION_PATH = ROOT_DIR / "submission.csv"

# The event types observed inside the train/test observation windows during
# research. Keeping the schema explicit makes train/test feature matrices
# deterministic and avoids dependence on crosstab column order.
EVENT_TYPES = [
    "item_view",
    "search_results_view",
    "photo_swipe",
    "favorite_add",
    "seller_page_view",
    "contact_phone_show",
    "login",
    "contact_chat_open",
    "contact_message_sent",
]

REQUIRED_TRAIN_COLUMNS = {
    "cookie_id",
    "cookie_created_at",
    "window_start_ts",
    "window_end_ts",
    "target",
}
REQUIRED_TEST_COLUMNS = {
    "cookie_id",
    "cookie_created_at",
    "window_start_ts",
    "window_end_ts",
}
REQUIRED_EVENT_COLUMNS = {
    "cookie_id",
    "event_ts",
    "eid",
    "event_name",
    "platform",
    "user_agent",
    "item_id",
    "item_category",
    "item_location",
    "seller_type",
    "search_query",
    "search_page",
    "pointer_x",
    "pointer_y",
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three input tables with the required datetime columns parsed."""
    for path in (TRAIN_PATH, TEST_PATH, EVENTS_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Required input file was not found: {path}"
            )

    train = pd.read_csv(
        TRAIN_PATH,
        parse_dates=[
            "cookie_created_at",
            "window_start_ts",
            "window_end_ts",
        ],
    )
    test = pd.read_csv(
        TEST_PATH,
        parse_dates=[
            "cookie_created_at",
            "window_start_ts",
            "window_end_ts",
        ],
    )
    events = pd.read_csv(
        EVENTS_PATH,
        parse_dates=["event_ts"],
    )

    return train, test, events


def validate_input(
        train: pd.DataFrame,
        test: pd.DataFrame,
        events: pd.DataFrame,
) -> None:
    """Validate the input schema and invariants required by the task."""
    missing_train = REQUIRED_TRAIN_COLUMNS - set(train.columns)
    missing_test = REQUIRED_TEST_COLUMNS - set(test.columns)
    missing_events = REQUIRED_EVENT_COLUMNS - set(events.columns)

    if missing_train:
        raise ValueError(f"Missing train columns: {sorted(missing_train)}")
    if missing_test:
        raise ValueError(f"Missing test columns: {sorted(missing_test)}")
    if missing_events:
        raise ValueError(f"Missing event columns: {sorted(missing_events)}")

    assert train["cookie_id"].is_unique
    assert test["cookie_id"].is_unique
    assert set(train["cookie_id"]).isdisjoint(set(test["cookie_id"]))

    assert train["target"].isin([0, 1]).all()
    assert (train["window_end_ts"] > train["window_start_ts"]).all()
    assert (test["window_end_ts"] > test["window_start_ts"]).all()

    assert (
            (train["window_end_ts"] - train["window_start_ts"])
            == pd.Timedelta(days=1)
    ).all()
    assert (
            (test["window_end_ts"] - test["window_start_ts"])
            == pd.Timedelta(days=1)
    ).all()


def events_in_window(
        events: pd.DataFrame,
        meta: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only events belonging to each cookie's half-open observation window."""
    meta_window = meta[
        ["cookie_id", "window_start_ts", "window_end_ts"]
    ]

    ev = events.merge(
        meta_window,
        on="cookie_id",
        how="inner",
        validate="many_to_one",
    )

    mask = (
            (ev["event_ts"] >= ev["window_start_ts"])
            & (ev["event_ts"] < ev["window_end_ts"])
    )

    ev = ev.loc[mask].copy()

    # Stable ordering is required for deterministic temporal differences when
    # timestamps are equal; eid is the event-level tie breaker used in research.
    ev = (
        ev.sort_values(
            ["cookie_id", "event_ts", "eid"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    ev["platform_normalized"] = (
        ev["platform"].astype(str).str.strip().str.lower()
    )

    return ev


def validate_events_in_window(
        ev_train: pd.DataFrame,
        ev_test: pd.DataFrame,
) -> None:
    """Assert that no event outside its cookie's observation window remains."""
    for ev in (ev_train, ev_test):
        assert (
                (ev["event_ts"] >= ev["window_start_ts"])
                & (ev["event_ts"] < ev["window_end_ts"])
        ).all()


def add_metadata_features(meta: pd.DataFrame) -> pd.DataFrame:
    """Build features available from cookie metadata before the event window."""
    result = meta[
        [
            "cookie_id",
            "cookie_created_at",
            "window_start_ts",
            "window_end_ts",
        ]
    ].copy()

    result["cookie_age_seconds"] = (
            result["window_start_ts"] - result["cookie_created_at"]
    ).dt.total_seconds()
    result["cookie_age_hours"] = result["cookie_age_seconds"] / 3600.0

    return result


def build_activity_features(ev: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event volume and behavioural diversity to cookie level."""
    g = ev.groupby("cookie_id", sort=False)

    features = pd.DataFrame(
        {
            "n_events": g.size(),
            "n_unique_items": g["item_id"].nunique(),
            "n_unique_categories": g["item_category"].nunique(),
            "n_unique_locations": g["item_location"].nunique(),
            "n_unique_search_queries": g["search_query"].nunique(),
            "n_unique_event_types": g["event_name"].nunique(),
            "n_unique_platforms": g["platform_normalized"].nunique(),
        }
    )

    return features.reset_index()


def build_event_count_features(ev: pd.DataFrame) -> pd.DataFrame:
    """Build per-cookie counts for each event type."""
    counts = pd.crosstab(ev["cookie_id"], ev["event_name"])
    counts = counts.reindex(columns=EVENT_TYPES, fill_value=0)
    counts.columns = [f"event_count__{event_type}" for event_type in EVENT_TYPES]
    return counts.reset_index()


def build_event_share_features(ev: pd.DataFrame) -> pd.DataFrame:
    """Build per-cookie event-type shares."""
    counts = pd.crosstab(ev["cookie_id"], ev["event_name"])
    counts = counts.reindex(columns=EVENT_TYPES, fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)
    shares = shares.fillna(0.0)
    shares.columns = [f"event_share__{event_type}" for event_type in EVENT_TYPES]
    return shares.reset_index()


def build_temporal_features(ev: pd.DataFrame) -> pd.DataFrame:
    """Aggregate timing, activity span and inter-event regularity features."""
    df = ev[
        [
            "cookie_id",
            "event_ts",
            "window_start_ts",
            "window_end_ts",
        ]
    ].copy()

    df["event_offset_seconds"] = (
            df["event_ts"] - df["window_start_ts"]
    ).dt.total_seconds()

    df["inter_event_seconds"] = (
        df.groupby("cookie_id", sort=False)["event_ts"]
        .diff()
        .dt.total_seconds()
    )

    df["relative_hour"] = (df["event_offset_seconds"] // 3600).astype(int)

    g = df.groupby("cookie_id", sort=False)

    features = g.agg(
        first_event_offset_seconds=("event_offset_seconds", "min"),
        last_event_offset_seconds=("event_offset_seconds", "max"),
        n_active_hours=("relative_hour", "nunique"),
        mean_inter_event_seconds=("inter_event_seconds", "mean"),
        median_inter_event_seconds=("inter_event_seconds", "median"),
        std_inter_event_seconds=("inter_event_seconds", "std"),
        min_inter_event_seconds=("inter_event_seconds", "min"),
        max_inter_event_seconds=("inter_event_seconds", "max"),
    ).reset_index()

    features["active_span_seconds"] = (
            features["last_event_offset_seconds"]
            - features["first_event_offset_seconds"]
    )

    n_events = g.size().rename("n_events").reset_index()
    features = features.merge(
        n_events,
        on="cookie_id",
        how="left",
        validate="one_to_one",
    )

    features["events_per_active_hour"] = features["n_events"] / (
            features["active_span_seconds"] / 3600.0 + 1.0
    )

    denominator = (
            features["std_inter_event_seconds"]
            + features["mean_inter_event_seconds"]
    )
    features["inter_event_burstiness"] = (
                                                 features["std_inter_event_seconds"]
                                                 - features["mean_inter_event_seconds"]
                                         ) / denominator.replace(0, np.nan)

    interval_cols = [
        "mean_inter_event_seconds",
        "median_inter_event_seconds",
        "std_inter_event_seconds",
        "min_inter_event_seconds",
        "max_inter_event_seconds",
        "inter_event_burstiness",
    ]
    features[interval_cols] = features[interval_cols].fillna(0.0)

    return features


def build_feature_table(
        meta: pd.DataFrame,
        ev: pd.DataFrame,
) -> pd.DataFrame:
    """Build the complete cookie-level feature table."""
    parts = [
        add_metadata_features(meta),
        build_activity_features(ev),
        build_event_count_features(ev),
        build_event_share_features(ev),
        build_temporal_features(ev),
    ]

    features = parts[0].copy()
    for part in parts[1:]:
        features = features.merge(
            part,
            on="cookie_id",
            how="left",
            validate="one_to_one",
        )

    return features


def canonicalize_features(
        X_train_raw: pd.DataFrame,
        X_test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Create one deterministic numeric feature schema for train and test."""
    train_df = X_train_raw.copy()
    test_df = X_test_raw.copy()

    # The experiments retained hours as the single canonical cookie-age feature.
    train_df = train_df.drop(columns=["cookie_age_seconds"])
    test_df = test_df.drop(columns=["cookie_age_seconds"])

    # n_events is computed both by activity and temporal aggregation. Keep one copy.
    n_events_candidates = [
        col for col in train_df.columns if col.startswith("n_events")
    ]
    if not n_events_candidates:
        raise ValueError("No n_events-derived feature is available")

    if "n_events" not in train_df.columns:
        source = n_events_candidates[0]
        train_df["n_events"] = train_df[source]
        test_df["n_events"] = test_df[source]
        n_events_candidates.append("n_events")

    duplicate_n_events = [
        col for col in n_events_candidates if col != "n_events"
    ]
    train_df = train_df.drop(columns=duplicate_n_events, errors="ignore")
    test_df = test_df.drop(columns=duplicate_n_events, errors="ignore")

    # Keep only numeric model features. The raw feature table also contains
    # timestamp metadata columns used to construct features, but CatBoost is
    # trained only on the numeric engineered representation used in research.
    feature_cols = [
        col
        for col in train_df.columns
        if col != "cookie_id"
           and pd.api.types.is_numeric_dtype(train_df[col])
    ]

    # Reindex test explicitly to the train feature schema.
    train_df = train_df[["cookie_id"] + feature_cols]
    test_df = test_df.reindex(columns=["cookie_id"] + feature_cols)

    missing_train_columns = set(feature_cols) - set(train_df.columns)
    missing_test_columns = set(feature_cols) - set(test_df.columns)
    if missing_train_columns or missing_test_columns:
        raise ValueError(
            "Feature schema mismatch: "
            f"missing_train={sorted(missing_train_columns)}, "
            f"missing_test={sorted(missing_test_columns)}"
        )

    return train_df, test_df, feature_cols


def fill_and_validate_features(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill aggregation NaNs for cookies without events and validate matrices."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Left joins preserve cookies with empty observation windows. For those cookies,
    # event-derived aggregates have no group row and therefore need zero defaults.
    X_train[feature_cols] = X_train[feature_cols].fillna(0.0)
    X_test[feature_cols] = X_test[feature_cols].fillna(0.0)

    assert list(X_train.columns) == ["cookie_id"] + feature_cols
    assert list(X_test.columns) == ["cookie_id"] + feature_cols
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert X_train[feature_cols].isna().sum().sum() == 0
    assert X_test[feature_cols].isna().sum().sum() == 0
    assert np.isfinite(X_train[feature_cols].to_numpy()).all()
    assert np.isfinite(X_test[feature_cols].to_numpy()).all()

    return X_train, X_test


def validate_feature_coverage(
        meta: pd.DataFrame,
        feature_table: pd.DataFrame,
) -> None:
    """Ensure feature engineering neither loses nor invents cookies."""
    meta_ids = set(meta["cookie_id"])
    feature_ids = set(feature_table["cookie_id"])
    assert meta_ids == feature_ids
    assert feature_table["cookie_id"].is_unique


def create_model() -> CatBoostClassifier:
    """Create the final model selected by walk-forward validation."""
    return CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        verbose=False,
        thread_count=-1,
        allow_writing_files=False,
    )


def build_submission(
        test: pd.DataFrame,
        scores: np.ndarray,
) -> pd.DataFrame:
    """Create and validate the exact two-column competition submission."""
    scores = np.asarray(scores, dtype=float)

    submission = pd.DataFrame(
        {
            "cookie_id": test["cookie_id"].to_numpy(),
            "score": scores,
        }
    )

    assert list(submission.columns) == ["cookie_id", "score"]
    assert len(submission) == len(test)
    assert submission["cookie_id"].is_unique
    assert submission["cookie_id"].notna().all()
    assert submission["score"].notna().all()
    assert np.isfinite(submission["score"].to_numpy()).all()
    assert submission["score"].between(0, 1).all()
    assert submission["cookie_id"].equals(test["cookie_id"])

    return submission


def main() -> None:
    train, test, events = load_data()
    validate_input(train, test, events)

    ev_train = events_in_window(events, train)
    ev_test = events_in_window(events, test)
    validate_events_in_window(ev_train, ev_test)

    print(f"Train: {train.shape}")
    print(f"Test:  {test.shape}")
    print(f"Events: {events.shape}")
    print(f"Train events in window: {len(ev_train)}")
    print(f"Test events in window:  {len(ev_test)}")

    X_train_raw = build_feature_table(train, ev_train)
    X_test_raw = build_feature_table(test, ev_test)

    validate_feature_coverage(train, X_train_raw)
    validate_feature_coverage(test, X_test_raw)

    X_train, X_test, feature_cols = canonicalize_features(
        X_train_raw,
        X_test_raw,
    )
    X_train, X_test = fill_and_validate_features(
        X_train,
        X_test,
        feature_cols,
    )

    assert "target" not in X_train.columns
    assert "target" not in X_test.columns
    assert "cookie_id" not in feature_cols

    y_train = train["target"].astype(int).to_numpy()

    model = create_model()
    model.fit(X_train[feature_cols], y_train)

    # No classification threshold is applied. The competition evaluates the
    # ranking of continuous scores and chooses its own threshold.
    scores = model.predict_proba(X_test[feature_cols])[:, 1]
    submission = build_submission(test, scores)
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"Features: {len(feature_cols)}")
    print(f"Submission: {SUBMISSION_PATH}")
    print(f"Submission shape: {submission.shape}")
    print(
        "Score range: "
        f"[{submission['score'].min():.6f}, "
        f"{submission['score'].max():.6f}]"
    )
    print("Submission validation: PASS")


if __name__ == "__main__":
    main()
