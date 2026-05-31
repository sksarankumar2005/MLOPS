"""GrievX public grievance resolution model.

This script follows the same step-by-step project flow as the Bangalore home
price notebook, but adapted for the grievance domain:
load data, inspect it, build features, train a Random Forest regressor,
evaluate it, and export reusable artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
COMPLAINTS_FILE = BASE_DIR / "GrievX _5 _Complaints_Dataset.xlsx"
MLA_FILE = BASE_DIR / "Final_MLA_and_Ward_member_Dataset.csv"
MINISTER_FILE = BASE_DIR / "tamilnadu_2021_ministers_dataset.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
JOBLIB_FILE = ARTIFACT_DIR / "grievx_resolution_model.joblib"
METADATA_FILE = ARTIFACT_DIR / "grievx_metadata.json"


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    complaints = pd.read_excel(COMPLAINTS_FILE)
    mla = pd.read_csv(MLA_FILE)
    ministers = pd.read_csv(MINISTER_FILE)
    return complaints, mla, ministers


def build_lookup_tables(
    complaints: pd.DataFrame,
    mla: pd.DataFrame,
    ministers: pd.DataFrame,
) -> tuple[dict[str, dict[str, str]], dict[str, list[dict[str, Any]]]]:
    category_lookup: dict[str, dict[str, str]] = {}
    for _, row in complaints.drop_duplicates(subset=["Problem_Category"]).iterrows():
        category_lookup[str(row["Problem_Category"]).strip().lower()] = {
            "Ministry": str(row["Ministry"]).strip(),
            "Minister_Name": str(row["Minister_Name"]).strip(),
        }

    district_lookup: dict[str, list[dict[str, Any]]] = {}
    for district, group in mla.groupby("district", dropna=False):
        district_lookup[str(district).strip().lower()] = (
            group[
                [
                    "ac_name",
                    "ac_no",
                    "district",
                    "winning_cand",
                    "MLA_Party",
                    "Panchayat_Union (Ooratchi)",
                    "Ward_Name",
                    "Ward_Member_Name",
                    "Ward_Member_Party",
                    "Governor",
                    "Chief Minister",
                ]
            ]
            .drop_duplicates()
            .to_dict(orient="records")
        )

    ministers_catalog = (
        ministers[["Minister_Name", "Ministry", "Responsibilities"]]
        .drop_duplicates()
        .fillna("")
        .to_dict(orient="records")
    )

    district_lookup["_minister_catalog"] = ministers_catalog  # type: ignore[assignment]
    return category_lookup, district_lookup


def build_model_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "complaint_text",
                TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1),
                "Problem_Name",
            ),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore"),
                ["Problem_Category"],
            ),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(random_state=42, n_jobs=-1)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_model(pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = pipeline.predict(x_test)
    return {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }


def tune_model(pipeline: Pipeline, x_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    parameter_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 4],
        "model__min_samples_leaf": [1, 2],
    }
    search = GridSearchCV(
        pipeline,
        parameter_grid,
        scoring="neg_mean_absolute_error",
        cv=5,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(x_train, y_train)
    return search


def predict_resolution_days(model: Pipeline, complaint_name: str, complaint_category: str) -> float:
    sample = pd.DataFrame(
        [
            {
                "Problem_Name": complaint_name,
                "Problem_Category": complaint_category,
            }
        ]
    )
    return float(model.predict(sample)[0])


def route_complaint(
    category_lookup: dict[str, dict[str, str]],
    district_lookup: dict[str, list[dict[str, Any]]],
    complaint_category: str,
    district: str,
) -> dict[str, Any]:
    category_key = complaint_category.strip().lower()
    district_key = district.strip().lower()
    route = {
        "category": complaint_category,
        "district": district,
        "ministry_info": category_lookup.get(category_key, {}),
        "district_hierarchy": district_lookup.get(district_key, []),
    }
    return route


def save_artifacts(best_model: Pipeline, metadata: dict[str, Any]) -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, JOBLIB_FILE)
    with METADATA_FILE.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def main() -> None:
    complaints, mla, ministers = load_datasets()

    print("=== Data Load ===")
    print("Complaints:", complaints.shape)
    print("MLA/Ward data:", mla.shape)
    print("Minister data:", ministers.shape)
    print()

    print("=== Data Cleaning ===")
    print("Missing values in complaints:\n", complaints.isna().sum())
    print("Complaint categories:", sorted(complaints["Problem_Category"].unique().tolist()))
    print("Districts covered in MLA dataset:", sorted(mla["district"].dropna().unique().tolist()))
    print()

    category_lookup, district_lookup = build_lookup_tables(complaints, mla, ministers)

    features = complaints[["Problem_Name", "Problem_Category"]].copy()
    target = complaints["Days_To_Complete"].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    print("=== Model Training ===")
    pipeline = build_model_pipeline()
    search = tune_model(pipeline, x_train, y_train)
    best_model = search.best_estimator_
    print("Best parameters:", search.best_params_)
    print("Best cross-validation score (neg MAE):", round(search.best_score_, 4))
    print()

    print("=== Model Evaluation ===")
    scores = evaluate_model(best_model, x_test, y_test)
    print("MAE:", round(scores["mae"], 3))
    print("R2:", round(scores["r2"], 3))
    print()

    print("=== Sample Predictions ===")
    examples = [
        ("Water leakage near street pipeline", "water"),
        ("Pothole on village main road", "roads"),
        ("Crop damage after rain", "agriculture"),
    ]
    for complaint_name, complaint_category in examples:
        predicted_days = predict_resolution_days(best_model, complaint_name, complaint_category)
        print(f"{complaint_name} -> {predicted_days:.2f} days")
    print()

    metadata = {
        "problem_categories": sorted(complaints["Problem_Category"].dropna().unique().tolist()),
        "problem_names": sorted(complaints["Problem_Name"].dropna().unique().tolist()),
        "districts_supported": sorted(mla["district"].dropna().unique().tolist()),
        "minister_lookup": category_lookup,
        "district_lookup": district_lookup,
        "model_file": JOBLIB_FILE.name,
    }
    save_artifacts(best_model, metadata)

    print("=== Export ===")
    print("Saved:", JOBLIB_FILE)
    print("Saved:", METADATA_FILE)
    print()

    route = route_complaint(category_lookup, district_lookup, "water", "Erode")
    print("=== Routing Example ===")
    print(json.dumps(route, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()