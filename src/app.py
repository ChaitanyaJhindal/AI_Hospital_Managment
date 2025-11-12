import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Streamlit setup
st.set_page_config(page_title="AI Hospital RM — Fuzzy + CSP", layout="wide")
st.title("🏥 AI-driven Hospital Resource Management System")

# Dataset path
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "hospital_patients_dataset.csv")
DATA_PATH = os.path.abspath(DATA_PATH)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# Load dataset
try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Dataset not found at {DATA_PATH}. Please place hospital_patients_dataset.csv in the data/ folder.")
    st.stop()

# Show basic info
st.subheader("📊 Dataset Snapshot")
st.write(f"Rows: {df.shape[0]} — Columns: {df.shape[1]}")
st.dataframe(df.head(8))

# Summary
st.subheader("📈 Basic Summary")
cols = st.multiselect("Columns to summarize", options=list(df.columns), default=["age","heart_rate","spo2","temperature"])
if cols:
    st.write(df[cols].describe().transpose())

# Histograms
st.subheader("📊 Vital Sign Distributions")
col1, col2, col3 = st.columns(3)
with col1:
    st.write("Heart Rate")
    st.bar_chart(df["heart_rate"].value_counts().sort_index().head(40))
with col2:
    st.write("SpO2")
    st.bar_chart(df["spo2"].value_counts().sort_index().head(40))
with col3:
    st.write("Temperature")
    st.bar_chart(df["temperature"].value_counts().sort_index().head(40))

# ---------------------- Step 1 ----------------------
st.markdown("---")
st.subheader("🧠 Step 1 — ML Baseline (RandomForest)")

target = st.selectbox("Choose target column", options=["severity_category", "recommended_bed_type"], index=0)
st.write("This will train a simple RandomForest model for demo purposes.")

if st.button("Run baseline ML training"):
    df2 = df.copy()
    numeric_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found for training.")
    else:
        X = df2[numeric_cols].fillna(0)
        y = df2[target].astype(str).fillna("unknown")
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        st.success(f"Baseline RandomForest accuracy: {acc:.3f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, preds, target_names=le.classes_))

# ---------------------- Step 2 ----------------------
st.markdown("---")
st.markdown("## Step 2 — 🤖 Fuzzy Logic Triage System")

import modules.fuzzy_triage as fuzzy_triage

if "df_processed" not in st.session_state:
    st.session_state["df_processed"] = df.copy()

if st.button("Compute fuzzy severity scores"):
    with st.spinner("Running fuzzy inference..."):
        system_ctrl = fuzzy_triage.build_fuzzy_system()
        required_cols = ["heart_rate", "spo2", "temperature", "respiratory_rate"]
        for c in required_cols:
            if c not in df.columns:
                st.error(f"Missing column: {c}")
                st.stop()

        df_proc = df.copy()
        df_proc["fuzzy_severity"] = df_proc.apply(lambda row: fuzzy_triage.compute_severity(row, system_ctrl), axis=1)
        st.session_state["df_processed"] = df_proc

    st.success("✅ Fuzzy severity scores computed!")
    st.dataframe(df_proc[["patient_id", "heart_rate", "spo2", "temperature", "respiratory_rate", "fuzzy_severity"]].head(10))
    st.metric("Average Severity (0–1)", f"{df_proc['fuzzy_severity'].mean():.3f}")
    st.bar_chart(df_proc["fuzzy_severity"])

# ---------------------- Step 3 ----------------------
st.markdown("---")
st.markdown("## Step 3 — 🧭 A* Search for Bed Allocation")

import modules.a_star_bed_allocation as bed_alloc

if st.button("Run A* Bed Allocation Optimization"):
    df_proc = st.session_state.get("df_processed", None)
    if df_proc is None or "fuzzy_severity" not in df_proc.columns:
        st.warning("⚠️ Please run 'Compute fuzzy severity scores' first (Step 2).")
        st.stop()

    if "patient_id" not in df_proc.columns:
        df_proc["patient_id"] = range(1, len(df_proc) + 1)

    with st.spinner("Allocating beds using A* search..."):
        patients_list = df_proc[["patient_id", "fuzzy_severity"]].to_dict(orient="records")
        allocations = bed_alloc.allocate_beds(patients_list)

    alloc_df = pd.DataFrame(allocations)
    alloc_df["Distance_Cost"] = pd.to_numeric(alloc_df["Distance_Cost"], errors="coerce")

    st.success("✅ Bed allocation complete!")
    st.dataframe(alloc_df)
    st.info("Each patient is assigned a bed based on severity using A* search optimization.")

# ---------------------- Step 4 ----------------------
st.markdown("---")
st.markdown("## Step 4 — 🕒 CSP-based Staff & Surgery Scheduling System")

import modules.csp_scheduler as csp

if st.button("Run CSP Scheduling Optimization"):
    with st.spinner("Solving scheduling constraints..."):
        df_proc = st.session_state.get("df_processed", df)
        patients = df_proc[["patient_id", "fuzzy_severity"]].to_dict(orient="records")

        schedule = csp.build_schedule(patients)

    if schedule:
        st.success("✅ Schedule generated successfully!")
        st.dataframe(pd.DataFrame(schedule))
    else:
        st.error("No feasible schedule found.")
