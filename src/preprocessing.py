"""Cleaning and leak-safe preprocessing for the COMPAS dataset."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_diagnostics import flag_invalid_values


def canonicalize_categories(
    df: pd.DataFrame, columns_and_maps: dict, placeholder_tokens: set
) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in columns_and_maps.items():
        cleaned = out[col].astype("string").str.strip()
        lowered = cleaned.str.lower()
        missing = out[col].isna() | lowered.isin(placeholder_tokens)
        out[col] = lowered.map(mapping).fillna(cleaned)
        out.loc[missing, col] = np.nan
    return out


def _fix_consistency_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Make derived categories agree with their valid numeric source columns."""
    out = df.copy()

    if {"age", "age_cat"}.issubset(out.columns):
        valid_age = out["age"].between(18, 100)
        out.loc[valid_age & (out["age"] < 25), "age_cat"] = "Less than 25"
        out.loc[valid_age & out["age"].between(25, 44), "age_cat"] = "25 - 45"
        # The supplied dataset consistently places age 45 in this group.
        out.loc[valid_age & (out["age"] >= 45), "age_cat"] = "Greater than 45"

    if {"decile_score", "score_text"}.issubset(out.columns):
        score = out["decile_score"]
        out.loc[score.between(1, 4), "score_text"] = "Low"
        out.loc[score.between(5, 7), "score_text"] = "Medium"
        out.loc[score.between(8, 10), "score_text"] = "High"

    return out


def clean_dataset(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """Apply deterministic cleaning rules found during exploratory analysis."""
    out = df.copy()
    placeholder_tokens = set(diagnostics_config.get("placeholder_tokens", []))

    for column in diagnostics_config.get("numeric_text_columns", []):
        if column in out.columns:
            cleaned = out[column].replace(list(placeholder_tokens), np.nan)
            out[column] = pd.to_numeric(cleaned, errors="coerce")

    flag_invalid_values(out, diagnostics_config.get("validity_rules", {}))
    out = canonicalize_categories(
        out,
        diagnostics_config.get("canonical_categories", {}),
        placeholder_tokens,
    )
    for column, value in diagnostics_config.get("constant_fill", {}).items():
        if column in out.columns:
            out[column] = out[column].fillna(value)
    out = _fix_consistency_issues(out)

    out = out.drop_duplicates()
    id_column = diagnostics_config.get("id_column")
    if id_column and id_column in out.columns:
        out = out.drop_duplicates(subset=id_column, keep="first")

    redundant = [
        column
        for column in diagnostics_config.get("redundant_columns", [])
        if column in out.columns
    ]
    return out.drop(columns=redundant)


def add_missingness_indicators(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Record whether selected values were missing before they are imputed."""
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[f"{column}_was_missing"] = out[column].isna().astype(int)
    return out


def split_features_target(
    df: pd.DataFrame, data_config: dict, missingness_indicators: list
):
    """Separate model features, target, and columns used only for auditing."""
    df = add_missingness_indicators(df, missingness_indicators)
    target = data_config["target"]
    sensitive_attr = data_config["sensitive_attr"]

    y = df[target]
    extras = df[[sensitive_attr, "score_text"]].copy()

    excluded = set(data_config.get("drop_columns", [])) | {target, sensitive_attr}
    feature_columns = [column for column in df.columns if column not in excluded]
    return df[feature_columns], y, extras


def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    """Build preprocessing that will be fitted on the training data only."""
    if preprocessing_config.get("encoder") != "onehot":
        raise ValueError("This pipeline currently supports encoder: onehot")

    scaler_name = preprocessing_config.get("scaler", "standard")
    if scaler_name not in {"standard", "none"}:
        raise ValueError("Scaler must be either standard or none")
    scaler = StandardScaler() if scaler_name == "standard" else "passthrough"
    imputation = preprocessing_config.get("imputation", {})

    numeric_pipeline = Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy=imputation.get("numeric_strategy", "median")),
            ),
            ("scale", scaler),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy=imputation.get(
                        "categorical_strategy", "most_frequent"
                    )
                ),
            ),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    indicator_columns = [
        f"{column}_was_missing"
        for column in preprocessing_config.get("mnar_indicator_sources", [])
    ]

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, preprocessing_config["numeric_features"]),
            (
                "categorical",
                categorical_pipeline,
                preprocessing_config["categorical_features"],
            ),
            ("indicators", "passthrough", indicator_columns),
        ]
    )


def split_train_test(
    X,
    y,
    extras,
    test_size: float,
    random_state: int,
):
    """Split features, target, and audit columns while keeping rows aligned."""
    return train_test_split(
        X,
        y,
        extras,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
