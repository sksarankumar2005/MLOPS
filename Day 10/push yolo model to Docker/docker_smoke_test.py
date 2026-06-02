"""Smoke tests for the Docker image.

This script verifies that the model artifacts exist, the Flask app imports,
the YOLO/text models load, and a couple of lightweight routes respond.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
EXPECTED_PATHS = [
    PROJECT_ROOT / "GrievX_UI_and_model" / "artifacts" / "grievx_resolution_model.joblib",
    PROJECT_ROOT / "GrievX_UI_and_model" / "artifacts" / "grievx_metadata.json",
    PROJECT_ROOT / "GrievX_UI_and_model" / "Final_MLA_and_Ward_member_Dataset.csv",
    PROJECT_ROOT / "runs" / "grievx_yolov8s_cpu" / "weights" / "best.pt",
]


def check_paths() -> None:
    missing_paths = [str(path) for path in EXPECTED_PATHS if not path.exists()]
    if missing_paths:
        raise FileNotFoundError("Missing required model files:\n- " + "\n- ".join(missing_paths))


def run_smoke_tests() -> None:
    check_paths()

    from app import PROBLEM_CATALOG, app as flask_app, predict_days, text_model, ward_df

    if ward_df.empty:
        raise RuntimeError("Ward dataset is empty")

    ward_sample = ward_df[["district", "ac_name", "Ward_Name"]].dropna().iloc[0]
    district = str(ward_sample["district"]).strip()
    area = str(ward_sample["ac_name"]).strip()
    ward_name = str(ward_sample["Ward_Name"]).strip()

    client = flask_app.test_client()

    response = client.get("/")
    if response.status_code != 200:
        raise RuntimeError(f"GET / failed with status {response.status_code}")

    category = next(iter(PROBLEM_CATALOG))
    problems = PROBLEM_CATALOG[category]
    if not problems:
        raise RuntimeError(f"No problems registered for category: {category}")

    problems_response = client.get(f"/get_problems/{category}")
    if problems_response.status_code != 200:
        raise RuntimeError(f"GET /get_problems/{category} failed with status {problems_response.status_code}")

    predict_response = client.post(
        "/predict_text",
        json={
            "district": district,
            "area": area,
            "ward": ward_name,
            "category": category,
            "problem_name": problems[0],
            "details": "Docker smoke test complaint",
        },
    )
    if predict_response.status_code != 200:
        raise RuntimeError(f"POST /predict_text failed with status {predict_response.status_code}")

    predict_payload = predict_response.get_json(silent=True) or {}
    if not predict_payload.get("success"):
        raise RuntimeError(f"POST /predict_text returned failure: {predict_payload}")

    predicted_days = predict_days(text_model, problems[0], category)
    if predicted_days <= 0:
        raise RuntimeError(f"Unexpected predicted days value: {predicted_days}")

    print("Docker smoke test passed.")
    print(f"Sample category: {category}")
    print(f"Sample district: {district}")
    print(f"Sample area: {area}")
    print(f"Sample ward: {ward_name}")
    print(f"Sample prediction days: {predicted_days:.2f}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run Docker smoke tests for GrievX Video.")
    parser.add_argument("--mode", default="local", choices=["local", "build", "runtime"], help="Test mode label")
    args = parser.parse_args(argv)

    print(f"Running smoke tests in {args.mode} mode...")
    run_smoke_tests()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))