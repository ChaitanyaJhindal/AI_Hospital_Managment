# 🏥 AI-driven Hospital Resource Management System

### 📘 Overview

This project is an intelligent **Hospital Resource Management System** that integrates multiple AI techniques — **Machine Learning**, **Fuzzy Logic**, **A\* Search**, and **Constraint Satisfaction Scheduling (CSP)** — into a unified interactive platform built using **Streamlit**.

It automates **patient triage**, **bed allocation**, and **staff scheduling**, helping hospitals optimize critical operations in real time.

---

## 🚀 Features

| Module                                           | Description                                                                                                                        | Techniques Used                         |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| **1️⃣ Data Preprocessing & ML Baseline** | Loads and analyzes hospital patient data, and trains a RandomForest model to predict severity or bed type.                         | `scikit-learn`, `pandas`, `numpy` |
| **2️⃣ Fuzzy Logic Triage System**        | Uses fuzzy inference on vital signs (heart rate, SpO₂, temperature, respiratory rate) to compute a patient severity score (0–1). | `scikit-fuzzy`                        |
| **3️⃣ A\* Bed Allocation System**        | Allocates patients to available hospital beds using an A\* search algorithm to minimize distance cost and prioritize severity.     | Custom heuristic +`numpy`             |
| **4️⃣ CSP-based Scheduling**             | Assigns doctors, rooms, and time slots for surgeries based on constraints like availability and non-overlapping times.             | `python-constraint`                   |
| **🖥 Streamlit Frontend**                  | Provides an easy-to-use dashboard for all modules with interactive buttons and real-time output.                                   | `streamlit`                           |

---

## 🧩 Project Structure


hospital-rm/

│

├── data/

│   └── hospital_patients_dataset.csv        # Input dataset

│

├── src/

│   ├── app.py                               # Main Streamlit application

│   ├── modules/

│   │   ├── fuzzy_triage.py                  # Fuzzy Logic model

│   │   ├── a_star_bed_allocation.py         # A* search bed allocator

│   │   └── csp_scheduler.py                 # CSP scheduling system

│

├── venv/                                    # Virtual environment (not included in Git)

├── README.md

└── requirements.txt


---
## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/hospital-rm.git
cd hospital-rm
---
2️⃣ Create a Virtual Environment

python -m venv venv

3️⃣ Activate Environment

venv\Scripts\Activate.ps1

4️⃣ Install Dependencies

pip install -r requirements.txt

5️⃣ Run the Application

python -m streamlit run src/app.py


## 📊 How It Works

### 🔹 Step 1 — ML Baseline

* Loads `hospital_patients_dataset.csv`
* Displays data snapshot & summary
* Trains a **RandomForestClassifier** on selected target (e.g., severity category)

### 🔹 Step 2 — Fuzzy Logic Triage

* Inputs: `heart_rate`, `spo2`, `temperature`, `respiratory_rate`
* Outputs: continuous **severity score** (0 → mild, 1 → critical)

### 🔹 Step 3 — A* Bed Allocation

* Allocates beds minimizing distance cost
* Prioritizes patients with higher severity first

### 🔹 Step 4 — CSP Scheduling

* Assigns  **doctors** ,  **rooms** , and **time slots**
* Ensures no overlap (doctor/room/time) and respects constraints
* Produces fast and feasible schedule (solves in <2 seconds)

## 🧩 Future Enhancements

* Integrate real-time patient monitoring sensors
* Use deep learning for severity prediction
* Connect to cloud hospital databases
* Add admin login & patient tracking system
