from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_FILE = ARTIFACT_DIR / "grievx_resolution_model.joblib"
METADATA_FILE = ARTIFACT_DIR / "grievx_metadata.json"
WARD_FILE = BASE_DIR / "Final_MLA_and_Ward_member_Dataset.csv"
HISTORY_DIR = BASE_DIR / "history"
HISTORY_FILE = HISTORY_DIR / "grievance_history.jsonl"


st.set_page_config(page_title="GrievX", page_icon="🧭", layout="wide")


@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_FILE)
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    return model, metadata


@st.cache_resource
def load_ward_data():
    return pd.read_csv(WARD_FILE)


def ensure_state() -> None:
    if "current_case" not in st.session_state:
        st.session_state.current_case = None
    if "latest_notice" not in st.session_state:
        st.session_state.latest_notice = ""


def normalize_key(value: str) -> str:
    return value.strip().lower()


def predict_days(model, complaint_name: str, complaint_category: str) -> float:
    sample = pd.DataFrame(
        [
            {
                "Problem_Name": complaint_name,
                "Problem_Category": complaint_category,
            }
        ]
    )
    return float(model.predict(sample)[0])


PROBLEM_CATALOG = {
    "agriculture": ["Crop damage complaint"],
    "industries": ["Industrial pollution complaint"],
    "roads": ["Road damage complaint"],
    "rural development": ["Village drainage issue"],
    "water": ["Drinking water shortage"],
}


def clean_value(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    return str(value).strip()


def append_jsonl(file_path: Path, payload: dict) -> None:
    HISTORY_DIR.mkdir(exist_ok=True)
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_chain(metadata: dict, ward_row: pd.Series, complaint_category: str) -> list[dict]:
    category_lookup = metadata.get("minister_lookup", {})
    minister_info = category_lookup.get(normalize_key(complaint_category), {})

    ward_member = clean_value(ward_row.get("Ward_Member_Name"))
    mla_name = clean_value(ward_row.get("winning_cand"))
    chief_minister = clean_value(ward_row.get("Chief Minister"))
    governor = clean_value(ward_row.get("Governor"))

    return [
        {"stage": "Ward Member", "name": ward_member},
        {"stage": "MLA", "name": mla_name},
        {"stage": "Minister", "name": minister_info.get("Minister_Name", "Unknown"), "ministry": minister_info.get("Ministry", "Unknown")},
        {"stage": "Chief Minister", "name": chief_minister},
        {"stage": "Governor", "name": governor},
    ]


def unique_sorted(series: pd.Series) -> list[str]:
    values = [clean_value(value) for value in series.dropna().tolist()]
    return sorted(dict.fromkeys(values))


def save_history_record(record: dict) -> None:
    append_jsonl(HISTORY_FILE, record)


def load_history(limit: int = 20) -> pd.DataFrame:
    if not HISTORY_FILE.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).tail(limit)


def main() -> None:
    if not MODEL_FILE.exists() or not METADATA_FILE.exists():
        st.error("Model artifacts are missing. Run grievx_public_grievance_model.py first.")
        st.stop()

    ensure_state()
    model, metadata = load_assets()
    ward_df = load_ward_data()

    st.markdown(
        """
        <style>
        .hero {
            padding: 1.5rem;
            border-radius: 1rem;
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #22c55e 100%);
            color: white;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='hero'><h1>GrievX</h1><p>AI-powered public grievance resolution for Tamil Nadu</p></div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.subheader("Raise a complaint")
        district_options = unique_sorted(ward_df["district"])
        district = st.selectbox("1. Select district", district_options)
        district_filtered = ward_df[ward_df["district"].astype(str).str.strip().str.lower() == normalize_key(district)]

        area_options = unique_sorted(district_filtered["ac_name"])
        area = st.selectbox("2. Select area", area_options)

        area_filtered = district_filtered[district_filtered["ac_name"].astype(str).str.strip().str.lower() == normalize_key(area)]
        ward_options = unique_sorted(area_filtered["Ward_Name"])
        ward_name = st.selectbox("3. Select ward", ward_options)

        ward_filtered = area_filtered[area_filtered["Ward_Name"].astype(str).str.strip().str.lower() == normalize_key(ward_name)]
        ward_row = ward_filtered.iloc[0] if not ward_filtered.empty else area_filtered.iloc[0]

        complaint_category = st.selectbox("4. Select complaint category", list(PROBLEM_CATALOG.keys()))
        complaint_name = st.selectbox("5. Select problem name", PROBLEM_CATALOG.get(complaint_category, []))
        complaint_text = st.text_area("Complaint details", placeholder="Optional: add extra details here", height=120)

        predict_clicked = st.button("Predict resolution and route", type="primary", use_container_width=True)
        if predict_clicked:
            predicted_days = predict_days(model, complaint_name, complaint_category)
            chain = build_chain(metadata, ward_row, complaint_category)
            st.session_state.current_case = {
                "problem_name": complaint_name,
                "details": complaint_text.strip(),
                "category": complaint_category,
                "district": district,
                "area": area,
                "ward_name": ward_name,
                "predicted_days": predicted_days,
                "chain": chain,
                "stage_index": 0,
            }
            st.session_state.latest_notice = ""

        current_case = st.session_state.current_case
        if current_case:
            st.markdown("### Prediction result")
            st.metric("Estimated resolution time", f"{current_case['predicted_days']:.1f} days")
            st.success(f"Complaint assigned to: {current_case['chain'][0]['stage']} - {current_case['chain'][0]['name']}")
            st.write(f"**District:** {current_case['district']}")
            st.write(f"**Area:** {current_case['area']}")
            st.write(f"**Ward:** {current_case['ward_name']}")

            st.markdown("### Escalation order")
            for idx, stage in enumerate(current_case["chain"], start=1):
                label = f"{idx}. {stage['stage']} - {stage['name']}"
                if stage.get("ministry"):
                    label += f" ({stage['ministry']})"
                st.write(label)

            st.markdown("### Complaint action")
            yes_col, no_col = st.columns(2)
            if yes_col.button("Yes, solved", use_container_width=True):
                record = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "problem_name": current_case["problem_name"],
                    "category": current_case["category"],
                    "district": current_case["district"],
                    "area": current_case["area"],
                    "ward_name": current_case["ward_name"],
                    "predicted_days": current_case["predicted_days"],
                    "status": "closed",
                    "current_stage": current_case["chain"][current_case["stage_index"]]["stage"],
                    "current_assignee": current_case["chain"][current_case["stage_index"]]["name"],
                }
                save_history_record(record)
                st.success("Complaint closed and saved to history.")
                st.session_state.current_case = None

            if no_col.button("No, not solved", use_container_width=True):
                next_stage_index = min(current_case["stage_index"] + 1, len(current_case["chain"]) - 1)
                current_case["stage_index"] = next_stage_index
                next_stage = current_case["chain"][next_stage_index]
                record = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "problem_name": current_case["problem_name"],
                    "category": current_case["category"],
                    "district": current_case["district"],
                    "area": current_case["area"],
                    "ward_name": current_case["ward_name"],
                    "predicted_days": current_case["predicted_days"],
                    "status": "escalated",
                    "current_stage": next_stage["stage"],
                    "current_assignee": next_stage["name"],
                }
                save_history_record(record)
                st.warning(f"Escalated to {next_stage['stage']} - {next_stage['name']}")
                if next_stage_index >= len(current_case["chain"]) - 1:
                    st.info("Complaint has reached the highest authority in the chain.")

        if st.session_state.latest_notice:
            st.info(st.session_state.latest_notice)

    with right:
        st.subheader("Project summary")
        st.info("This interface predicts resolution time and routes complaints using district, area, and ward selection.")
        st.write("**Supported categories:**", ", ".join(PROBLEM_CATALOG.keys()))
        st.write("**Supported districts:**", ", ".join(unique_sorted(ward_df["district"])))

        st.markdown("### Workflow")
        st.code("District -> Area -> Ward -> Problem Category -> Problem Name -> Predict", language="text")

        st.markdown("### Escalation")
        st.write("If the complaint stays unresolved beyond the predicted deadline, it can be escalated automatically.")

        history_df = load_history()
        if not history_df.empty:
            st.markdown("### Recent history")
            safe_columns = [column for column in ["timestamp", "problem_name", "category", "district", "predicted_days", "current_stage", "current_assignee", "status"] if column in history_df.columns]
            st.dataframe(history_df[safe_columns], use_container_width=True)


if __name__ == "__main__":
    main()