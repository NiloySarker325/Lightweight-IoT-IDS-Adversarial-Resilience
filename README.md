# Evaluating Adversarial Attack Resilience in Lightweight ML Intrusion Detection Systems for IoT Networks

**Author:** Niloy Sarker Bappy

**Institution:** Department of Computer Information Science, Minnesota State University, Mankato, MN, USA

## Overview

The rapid expansion of Internet of Things (IoT) ecosystems requires the deployment of lightweight Machine Learning (ML) Intrusion Detection Systems (IDS) capable of operating on resource-constrained edge gateways. While shallow neural networks and decision tree ensembles achieve near-perfect detection rates on clean traffic, they remain critically vulnerable to domain-constrained adversarial evasion.

This repository contains the complete experimental pipeline for evaluating baseline performance, adversarial vulnerability, and defensive recovery of four lightweight classifiers (Decision Tree, Random Forest, XGBoost, and a 3-layer Multilayer Perceptron) using a balanced 100,000-flow subset from the **CICIoT2023** dataset.

## Key Findings

* **Baseline Feasibility:** All tested lightweight models achieved >96.5% clean accuracy with sub-second batch inference latencies (2.1 ms to 107.5 ms per 20,000 flows) on CPU-bound edge hardware.
* **White-Box Vulnerability (FGSM):** A Fast Gradient Sign Method ($\epsilon = 0.1$) attack degraded the 3-layer PyTorch MLP accuracy by ~11.4%.
* **Black-Box Vulnerability (HopSkipJump):** Without requiring internal model access, decision-based boundary probing degraded the Random Forest from 100.0% accuracy down to 42.0%–52.0%.
* **Defense Divergence:** Reactive **Feature Squeezing** (4-bit depth) proved counterproductive for high-precision tabular network telemetry, inducing a further -7.38% loss in accuracy. Proactive **Adversarial Training** completely neutralized the FGSM threat, recovering +11.58% (98.18% defended accuracy) with only 0.69 seconds of retraining overhead.

## Repository Structure

```text
├── codes/
│   ├── IoT_research.py             # Main pipeline: Preprocessing, Baseline, Attacks, and Defenses
│   ├── generate_all_figures.py     # Matplotlib script to generate publication-ready charts
│   └── file_type.py                # Dataset partition inspection utility
├── Data Sets/                      # Ignored in git (See Installation below)
│   └── MERGED_CSV/                 # Downloaded CICIoT2023 CSV partitions
├── figures/                        # 300 DPI visualization artifacts
│   ├── Figure_1_iot_research_results.png
│   ├── Figure_2_Baseline_Comparison.png
│   ├── Figure_3_HopSkipJump_MultiRun.png
│   ├── Figure_4_Defense_Tradeoffs.png
│   └── Figure_5_Latency_Overhead_LogScale.png
├── Evaluating Adversarial Attack Resilience... .docx  # Full IEEE-formatted research paper
└── README.md

```

## Installation & Setup

**1. Clone the repository:**

```bash
git clone https://github.com/YourUsername/Lightweight-IoT-IDS-Adversarial-Resilience.git
cd Lightweight-IoT-IDS-Adversarial-Resilience

```

**2. Install dependencies:**
This project requires Python 3.10+ and the following core libraries:

```bash
pip install pandas numpy scikit-learn xgboost torch torchvision matplotlib adversarial-robustness-toolbox

```

**3. Dataset Configuration:**
Due to GitHub's file size limits, the `Data Sets/` folder is excluded from this repository via `.gitignore`.

* Download the [CICIoT2023 Dataset](https://www.unb.ca/cic/datasets/iotdataset-2023.html).
* Extract the `.csv` partitions (e.g., `Merged01.csv`, `Merged02.csv`) into a local directory named `Data Sets/MERGED_CSV/` at the root of the project.

## Usage

**Running the Core Experimental Pipeline:**
The primary script executes the data extraction, Min-Max scaling, baseline model training, ART-wrapped evasion attacks, and defensive countermeasures.

```bash
python codes/IoT_research.py

```

*(Note: Ensure your terminal directory is set to the repository root before executing so relative paths resolve correctly).*

**Generating Visualizations:**
To recreate the high-resolution 300 DPI figures used in the research paper:

```bash
python codes/generate_all_figures.py

```

## Hardware Environment

All code was tested on a CPU-bound, edge-representative hardware profile to accurately simulate IoT gateway resource constraints:

* **Processor:** Intel Core Ultra 5 235U (2.00 GHz)
* **RAM:** 16.0 GB (5600 MT/s)
* **Execution:** Standard multi-threading (`n_jobs=-1` for Scikit-Learn tree ensembles) without dedicated datacenter GPU acceleration.