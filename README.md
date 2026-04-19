# ML-Based Traffic Signal Control System

This project implements an Adaptive Traffic Signal Control system using Reinforcement Learning (RL), focused on optimizing intersection performance in urban road networks. The system leverages the SUMO microscopic traffic simulator and the `sumo-rl` framework, applying a multi-agent Linear Actor-Critic algorithm for real-time adaptation to traffic conditions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Architecture](#architecture)
- [Libraries Used](#libraries-used)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

Managing urban traffic congestion is a crucial challenge in smart cities. This project uses RL to dynamically adjust signal phases at traffic intersections based on observed traffic states, aiming to minimize waiting time, queue length, and vehicle delay.

**Key Features:**
- Multi-agent Linear Actor-Critic algorithm for traffic control.
- Integration with SUMO traffic simulation via Python bindings (`sumo-rl`).
- Live dashboard visualization of training statistics using Matplotlib.
- Modular codebase, easily extensible for new environments or agents.

---

## Directory Structure

```plaintext
ML_based_traffic_system/
│
├── agent/
│   ├── linear_actor_critic.py   # Core RL agent implementation
│   └── multi_agent.py           # Multi-agent controller logic
│
├── baselines/
│   └── run_baselines.py         # Scripts for baseline methods (e.g., fixed-time, random)
│
├── dashboard/
│   └── (Live Matplotlib dashboard for training visualization)
│
├── data/
│   └── (Network and route XML files for SUMO simulation)
│
├── environment/
│   ├── observation_fn.py        # State extraction from SUMO
│   └── reward_fn.py             # Custom episode reward definitions
│
├── weights/                     # (Will be created to store trained agent models)
├── results/                     # (Will be created to store plots, metrics)
│
├── train.py                     # Entry point for training agents
├── evaluate.py                  # Evaluation scripts for trained policies
├── check_setup.py               # SUMO and environment checks
├── requirements.txt             # Python dependencies
├── .gitignore
├── test_sumo_cmd.py
├── _step_src.txt
├── _sumo_step_src.txt
└── README.md                    # [You are here]
```

---

## Architecture

- **Environment:** Wraps SUMO simulation using `sumo-rl`, supporting multiple intersections and extracting the current state (queue, wait times, etc.).
- **Agent:** Adopts a Linear Actor-Critic RL approach, learning optimal signal actions for intersections from traffic features.
- **Multi-Agent Controller:** Distributes an agent to each intersection; coordinates learning and action application.
- **Dashboard:** Live Matplotlib visualization tracks key metrics: episode reward, mean queue, wait, and delay.
- **Baselines:** Includes scripts for comparison to standard, non-learning traffic signal control strategies.
- **Data:** SUMO XML network/route files for the simulated scenario (e.g., Bremen network).

---

## Libraries Used

From [`requirements.txt`](https://github.com/Sidd623-exe/ML_based_traffic_system/blob/main/requirements.txt):

- **[sumo-rl](https://github.com/LucasAlegre/sumo-rl)** (>=2.0.0): RL interface for SUMO.
- **[gymnasium](https://www.gymnasium.farama.org/)** (>=0.28.0): RL environment API (successor to OpenAI Gym).
- **numpy** (>=1.24.0): Array, math, and statistics.
- **pandas** (>=2.0.0): Tabular data management.
- **scipy** (>=1.10.0): Additional scientific utilities.
- **matplotlib** (>=3.7.0): Plotting and live dashboard.
- **tqdm** (>=4.65.0): Progress bar for training loops.
- **tabulate** (>=0.9.0): For formatted tables.
- **SUMO Traffic Simulator**: Must be installed separately from [sumo.dlr.de](https://www.eclipse.org/sumo/).
  - Ensure SUMO_HOME is set and accessible to Python.

---

## Getting Started

### Prerequisites

- Python 3.8+
- SUMO Traffic Simulator ([installation guide](https://sumo.dlr.de/docs/Installing.html))
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Sidd623-exe/ML_based_traffic_system.git
cd ML_based_traffic_system

# Install Python dependencies
pip install -r requirements.txt

# Ensure SUMO and SUMO_HOME are set up
python check_setup.py
```

---

## Usage

### Training

```bash
# Headless training, 200 episodes
python train.py

# With SUMO GUI
python train.py --gui

# Quick smoke-test
python train.py --episodes 5 --seconds 300
```

### Evaluation

```bash
python evaluate.py
```

---

## Acknowledgments

- **SUMO**: Eclipse SUMO simulator ([official site](https://www.eclipse.org/sumo/))
- **sumo-rl**: Gym-compatible RL environments for SUMO ([repo](https://github.com/LucasAlegre/sumo-rl))

---

> This repository was developed for practical research in adaptive traffic signal control using interpretable reinforcement learning methods. Contributions and suggestions are welcome!
