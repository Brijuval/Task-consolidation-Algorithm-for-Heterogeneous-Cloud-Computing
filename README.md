# Energy-Aware Task Consolidation (EATC) in Heterogeneous Cloud Computing ☁️🌱

[![Java](https://img.shields.io/badge/Java-1.8%2B-ED8B00?style=for-the-badge&logo=java&logoColor=white)](https://www.java.com/)
[![CloudSim](https://img.shields.io/badge/CloudSim-3.0.3-blue?style=for-the-badge)](http://www.cloudbus.org/cloudsim/)
[![Energy Savings](https://img.shields.io/badge/Energy_Reduction-~23%25-brightgreen?style=for-the-badge)](#results--performance)

## 📌 About the Project
Built as a final-year Information Technology engineering capstone specializing in cloud computing infrastructure, this project tackles the critical issue of high energy consumption in modern data centers. 

Standard cloud schedulers (like Random Allocation or MaxUtil) often suffer from "blindness to heterogeneity"—assigning heavy computational tasks to slow servers. This keeps servers running at peak power for extended periods. The **EATC Algorithm** solves this using a predictive, deterministic approach that maps tasks to the fastest execution paths, minimizing active CPU time and maximizing server "Sleep Time."

## ✨ Key Features
* **Real-Time Scheduling:** Utilizes a Min-Heap data structure to achieve an $O(1)$ decision-making time, vastly outperforming slow meta-heuristic algorithms (like PSO or Ant Colony).
* **Heterogeneity-Aware:** Dynamically builds an **Expected Time to Compute (ETC) Matrix** to account for different VM speeds (MIPS) and Task sizes (MI).
* **"Race-to-Sleep" Strategy:** Prioritizes execution speed over mere resource packing, ensuring hardware enters power-saving idle states sooner.
* **Stochastic Workload Simulation:** Tested against dynamic, synthetically generated Poisson-arrival workloads to mimic unpredictable real-world traffic.

## ⚙️ The Algorithm Logic (How it Works)
1. **Intercept:** The custom `EATCBroker` intercepts incoming cloudlets.
2. **Matrix Calculation:** For every Task-VM pair, it calculates the estimated time: `Time = Task Length (MI) / VM Speed (MIPS)`.
3. **Min-Heap Sort:** All possible execution times are fed into a Min-Heap priority queue.
4. **Optimal Binding:** The broker extracts the smallest time value and instantly binds the task to that specific VM, ensuring the absolute shortest execution duration.

## 📊 Results & Performance
The algorithm was tested using **CloudSim 3.0.3** by simulating 100 heterogeneous tasks across 10 mixed-capacity virtual machines. The results were compared against industry-standard baselines utilizing a linear power model ($P(u) = k \cdot P_{max}$).

| Algorithm | Strategy | Total Energy Consumed | Performance |
| :--- | :--- | :--- | :--- |
| **Random Allocation** | Blind placement | ~265.50 kWh | Baseline (Worst) |
| **MaxUtil (Best Fit)**| Space / Resource Packing | ~241.80 kWh | Average |
| **Proposed EATC** | Time / Speed Optimization | **203.75 kWh** | **Best (~23% Savings)** |

*By finishing tasks earlier, EATC successfully reduced total data center energy consumption by over 20% without degrading system throughput.*

## 💻 Experimental Setup
* **Simulation Toolkit:** CloudSim 3.0.3
* **Datacenter:** 1 Heterogeneous Datacenter
* **Hosts:** 10 Hosts (Mix of High-Performance 3000 MIPS and Power-Efficient 1500 MIPS nodes)
* **Workload:** 100 Synthetic Cloudlets (Varying lengths from 1,000 to 10,000 MI)
* **Power Model:** 250W Peak Power, 60% Idle Power

## 🚀 How to Run Locally

### Prerequisites
* Java Development Kit (JDK) 8 or higher.
* Eclipse IDE (or IntelliJ IDEA).
* [CloudSim 3.0.3 JAR](https://github.com/Cloudslab/cloudsim/releases) files.

### Setup Instructions
1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/yourusername/EATC-Cloud-Scheduler.git](https://github.com/yourusername/EATC-Cloud-Scheduler.git)


📁 Repository Structure
├── src/
│   ├── org.cloudbus.cloudsim.examples/
│   │   ├── EATCBroker.java        # Core EATC algorithm logic (Min-Heap & ETC)
│   │   └── EATCSimulation.java    # Datacenter setup & main execution
├── docs/
│   ├── energy_comparison_chart.png # Performance graphs
│   └── capstone_presentation.pptx  # Project slides
├── README.md
└── .gitignore                     # Ignores compiled /bin files