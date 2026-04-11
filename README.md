# Task Consolidation Algorithm for Heterogeneous Cloud Computing

<div align="center">

**PMEC-X: Predictive Multifactor Energy Consolidation Scheduler**

[![Python](https://img.shields.io/badge/Python-3.12+-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Simulation](https://img.shields.io/badge/Simulation-24hr%20%7C%204150%20tasks-orange?style=flat-square)]()

</div>

---

## Paper Summary

PMEC-X is a heterogeneous cloud task consolidation algorithm designed to reduce idle power waste in data centers while preserving SLA compliance. The scheduler combines next-epoch workload prediction, urgency-based task ordering, 7-factor VM scoring, and SLA-tier-aware consolidation into a single scheduling loop that runs every 120 seconds.

### Abstract

Cloud data centers consume a significant share of global electricity, and a large part of that draw comes from underutilised virtual machines. PMEC-X addresses this by combining EWMA+CUSUM workload prediction with carbon-aware and thermal-aware scheduling, then applying a consolidation policy that can retain, migrate, queue, or shut down VMs depending on SLA urgency. In the 24-hour simulation used in this repository, PMEC-X reduced energy use to 74.60 kWh, achieved zero SLA violations, and detected 30 regime changes in one epoch each.

### Results — 24-Hour Simulation (4,150 tasks · 20 VMs · 720 epochs)

| Scheduler | Energy (kWh) | Violations | Viol % | Avg VMs | Shutdowns |
|---|---|---|---|---|---|
| **PMEC-X** | **74.60** | **0** | **0.0%** | **16.7** | **4** |
| MinMin | 87.53 | 0 | 0.0% | 20.0 | 0 |
| Round Robin | 125.10 | 294 | 7.1% | 20.0 | 0 |
| FCFS | 119.88 | 10 | 0.2% | 20.0 | 0 |
| MaxMin | 121.44 | 187 | 4.5% | 20.0 | 0 |

> PMEC-X saves 14.8% energy vs MinMin and 40.4% vs Round Robin while maintaining zero SLA violations.

---

## Core Contributions

| # | Contribution | Description |
|---|---|---|
| **C1** | Formal problem definition | Heterogeneous cloud task consolidation with seven normalised scoring factors, including carbon intensity and server thermal state |
| **C2** | EWMA+CUSUM hybrid prediction | Next-epoch VM load forecasting with a theoretical MSE bound and regime-change detection |
| **C3** | SLA-tier-aware consolidation | Differentiates CRITICAL, STANDARD, and BULK tasks to enable aggressive consolidation without SLA degradation |
| **C4** | Comparative simulation framework | Benchmarks PMEC-X against Round Robin, MinMin, FCFS, and MaxMin on a 4,150-task, 24-hour workload |

---

## How It Works

PMEC-X runs a 4-phase scheduling loop every 120 seconds:

```
INPUT: Task Queue → VM Pool → Carbon + Thermal Signals
         │
         ▼
 PHASE 1: EWMA + CUSUM Prediction
         Forecast next-epoch VM load and detect regime changes
         │
         ▼
 PHASE 2: Urgency Sort
         Rank tasks by priority and deadline slack
         U(τ) = 0.6 × Priority + 0.4 × Deadline Slack
         │
         ▼
 PHASE 3: 7-Factor Scoring Engine
         S(τ, v) = Σ wₖ · fₖ for every (task, VM) pair
         Speed · Headroom · Cost · Energy · Carbon · Thermal · Urgency
         │
         ▼
 PHASE 4: SLA-Tier-Aware Consolidation
         VM load < 12% → RETAIN / MIGRATE / QUEUE / SHUTDOWN
         │
         ▼
OUTPUT: Schedule · Energy Log · SLA Report · CUSUM Events
```

---

## 📁 Project Structure

```
PMECX/
├── core/
│   ├── models.py          # Task, VM, Container, Host data models
│   ├── predictor.py       # EWMA + CUSUM hybrid workload predictor
│   └── scorer.py          # 7-factor weighted scoring engine
│
├── simulation/
│   ├── workload.py        # Poisson workload generator (4,150 tasks/24hr)
│   └── environment.py     # Full scheduling loop + all baseline schedulers
│
├── baselines/
│   ├── roundrobin.py      # Round Robin baseline
│   ├── fcfs.py            # First Come First Served baseline
│   ├── minmin.py          # MinMin baseline
│   └── maxmin.py          # MaxMin baseline
│
├── dashboard/
│   └── app.py             # Live Streamlit dashboard
│
├── results/               # Simulation outputs (CSV)
└── run_simulation.py      # One-command runner
```

---

## ⚙️ Installation

**Prerequisites:** Python 3.10+, Git

```bash
# Clone the repository
git clone https://github.com/varun222004/PMECx.git
cd PMECx

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

# Install dependencies
pip install streamlit plotly matplotlib pandas numpy
```

---

## ▶️ Running the Project

### 1. Run the full 24-hour simulation

```bash
python run_simulation.py
```

Expected output:
```
PMEC-X        74.60 kWh    0.0%   16.7 VMs   4 shutdowns
Round Robin  125.10 kWh    7.1%   20.0 VMs   0 shutdowns
MinMin        87.53 kWh    0.0%   20.0 VMs   0 shutdowns
FCFS         119.88 kWh    0.2%   20.0 VMs   0 shutdowns

Energy saved: +14.8% vs MinMin | +40.4% vs Round Robin
CUSUM regime changes: 30
```

### 2. Launch the live dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`. Press **▶ Start** to watch all 4 schedulers run simultaneously in real time.

---

## 📊 Live Dashboard Features

| Feature | Description |
|---|---|
| Real-time energy chart | All 4 schedulers plotted live — watch PMEC-X stay low |
| VM heatmap | Live load per VM — see consolidation happening |
| CUSUM alarm box | Turns red when a regime change is detected |
| Carbon intensity dial | Shows the per-rack carbon signal used by the scorer |
| Weight sliders | Adjust all 7 scoring weights live |
| Spike injection | Simulate a sudden traffic burst — watch CUSUM fire |
| Results summary | Final per-scheduler metrics at the end of the run |

---

## 🔬 Algorithm Details

### EWMA + CUSUM Hybrid Predictor

```
EWMA:  û(t+1) = 0.3 · u(t) + 0.7 · û(t)
              MSE bound: 0.053 · σ²

CUSUM: C_high(t+1) = max(0, C_high(t) + deviation − k)
              Alarm when C_high > h = 0.3
```

- **EWMA** tracks smooth load in stable regimes
- **CUSUM** detects sudden spikes and fires in **1 epoch** vs 5–8 for EWMA alone
- Detected **30 regime changes** across the 24-hour simulation trace

### 7-Factor Scoring Function

```
S(τᵢ, vⱼ) = w₁·f_speed + w₂·f_headroom + w₃·f_cost + w₄·f_energy
           + w₅·f_carbon + w₆·f_thermal + w₇·f_urgency

All factors normalised to [0, 1] · Equal weights (1/7) by default
```

### Consolidation Decision Tree

```
VM load < 12% ?
├── Task = CRITICAL or STANDARD → RETAIN (never move)
├── Task = BULK + slack ≥ migration cost → MIGRATE (live migration)
├── Task = BULK + slack ≥ 0 → QUEUE (defer safely)
└── VM now empty → SHUTDOWN (idle power eliminated)
```

### Complexity

| Phase | Operation | Cost |
|---|---|---|
| Prediction | EWMA + CUSUM per VM | O(m) |
| Urgency sort | Quicksort | O(n log n) |
| Assignment | n tasks × m VMs | O(n·m) |
| Consolidation | Per VM per task | O(n + m) |
| **Total** | Dominated by assignment | **O(n·m + n log n)** |

Completes each epoch in **< 1ms** for n=4,150 tasks, m=20 VMs.

---

## 📈 Simulation Configuration

| Parameter | Value |
|---|---|
| VM pool | 20 VMs across 4 types (fast/ standard/ cheap/ hot-aisle) |
| MIPS ranges | fast: 8,000–10,000; standard: 3,000–6,000; cheap: 800–2,000; hot-aisle: 4,000–7,000 |
| Task count | 4,150 tasks over 24 hours |
| Arrival model | Non-homogeneous Poisson (λ = 3–18 tasks/epoch) |
| SLA tiers | 10% CRITICAL · 30% STANDARD · 60% BULK |
| Epoch duration | 120 seconds · 720 epochs = 24 hours |
| EWMA α | 0.3 |
| CUSUM k / h | 0.05 / 0.3 |
| Consolidation threshold | 0.12 (θ_low) |
| Carbon scores | Rack A 0.25 · Rack B 0.50 · Rack C 0.45 · Rack D 0.75 |
| Random seed | 42 (fully reproducible) |

---

## 🌱 SDG Alignment

- **SDG 7** — Affordable and Clean Energy: reduces data center electricity consumption and carbon emissions
- **SDG 9** — Industry, Innovation and Infrastructure: novel algorithm contribution to sustainable cloud computing

---

## 🔮 Future Work

- **Live Carbon API Integration** — real-time gCO₂/kWh signals via WattTime or Electricity Maps
- **Bayesian Adaptive Weight Tuning** — Gaussian Process bandit optimisation of the weight vector W
- **Multi-Datacenter Scheduling** — add network latency as an eighth scoring factor
- **CloudSim Plus Validation** — benchmark against Google Cluster Trace 2019
- **Kubernetes Integration** — production deployment of the container-layer scheduling component

---

## 📚 References

1. Beloglazov, A., Abawajy, J., & Buyya, R. — *Energy-aware resource allocation heuristics*, FGCS 2012
2. Braun, T. D. et al. — *Comparison of eleven static heuristics for heterogeneous task mapping*, JPDC 2001
3. Fan, X., Weber, W. D., & Barroso, L. A. — *Power provisioning for a warehouse-sized computer*, ISCA 2007
4. Calheiros, R. N. et al. — *CloudSim: A toolkit for cloud computing simulation*, SPE 2011
5. Page, E. S. — *Continuous inspection schemes*, Biometrika 1954
6. Xu, M., Tian, W., & Buyya, R. — *Survey on load balancing algorithms for VM placement*, C&C 2017

---

## 👨‍💻 Author

**Authors**

- Varun V — Dept. of Information Technology, Alliance University
- Valmeeki Singh — Dept. of Information Technology, Alliance University
- Santhosh G — Dept. of Information Technology, Alliance University
- Abdul Wahab — Dept. of Information Technology, Alliance University

Repository maintainer: Varun

---

<div align="center">

**⚡ PMEC-X — Predict. Consolidate. Save.**

*Built from scratch in Python. No cloud account needed. Runs entirely on your laptop.*

</div>
