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
| **5️⃣ RAG AI Summarizer** 🤖             | Converts technical results into simple English summaries using Groq's LLM API (Llama 3.3 70B model).                              | `groq`, Llama 3.3 70B                |
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

│   ├── algorithm_comparison.ipynb           # Jupyter notebook comparing A*, BFS, DFS, Greedy

│   ├── test_rag.py                          # Test script for RAG summarizer

│   ├── modules/

│   │   ├── fuzzy_triage.py                  # Fuzzy Logic model

│   │   ├── a_star_bed_allocation.py         # A* search bed allocator

│   │   ├── csp_scheduler.py                 # CSP scheduling system

│   │   └── rag_summarizer.py                # RAG AI summarizer using Groq API

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

### 🔹 Step 5 — RAG AI Summarizer 🤖

* **Powered by Groq API** with Llama 3.3 70B model
* Converts A* and CSP results into **simple English summaries**
* Provides actionable insights for hospital staff
* Three summary types:
  - Bed allocation summary
  - Surgery schedule summary
  - Comprehensive combined report

---

## 🤖 RAG AI Summarizer

### Overview
The RAG (Retrieval Augmented Generation) module uses **Groq's LLM API** to generate human-friendly summaries of technical results.

### How It Works
```
Algorithm Results → RAG Summarizer → Groq API (Llama 3.3 70B) → Simple English Summary
```

### Usage in App
1. Run fuzzy triage (Step 2)
2. Run A* bed allocation (Step 3) → See AI summary
3. Run CSP scheduling (Step 4) → See AI summary
4. Click **"Generate Complete AI Summary"** → Comprehensive report

### Example Output
```
We had 20 patients who needed beds in the hospital, and we were able to 
give each of them a bed. Patients were prioritized based on how sick they 
are, with the sickest patients getting beds closest to the care they need.

The "distance cost" refers to how far the patient's bed is from the care 
they need. A distance of 0 means the patient is right next to the care 
they need, which is ideal.
```

### Standalone Usage
```python
from modules.rag_summarizer import summarize_results

summary = summarize_results(
    allocation_data=allocations,
    schedule_data=schedule,
    total_patients=20
)
print(summary)
```

### API Configuration
- **Model**: `llama-3.3-70b-versatile`
- **Temperature**: 0.7
- **API Key**: Configured in `rag_summarizer.py`

---

## 📊 Algorithm Comparison Notebook

### `algorithm_comparison.ipynb`
Interactive Jupyter notebook comparing bed allocation algorithms:
- **A\* Search** - Optimal & efficient ✅
- **BFS** - Optimal but slower
- **DFS** - Fast but suboptimal
- **Greedy Best-First** - Fast but suboptimal

### Features
- Performance benchmarks (time, nodes explored, distance)
- 6+ visualizations showing why A* is best
- Comprehensive 4-in-1 dashboard
- Hospital bed allocation heatmap
- Detailed analysis and conclusions

### Run the Notebook
```bash
jupyter notebook src/algorithm_comparison.ipynb
```

---

## 🧩 Future Enhancements

* Integrate real-time patient monitoring sensors
* Use deep learning for severity prediction
* Connect to cloud hospital databases
* Add admin login & patient tracking system
