# Reinforcement Learning for Active Flow Control

This repository contains the full implementation of a master’s thesis project focused on **active flow control** in 2D incompressible fluid dynamics using **deep reinforcement learning (DRL)**. The goal is to reduce the recirculation area behind flow obstacles using jet actuators, controlled via DRL policies.

---

## Test Cases

The project features two benchmark geometries:

- **Backward-Facing Step (BFS)**
- **Cylinder with Wall Jets**

Each environment is modeled using the [FEniCS Project (2019.1.0)](https://fenicsproject.org/), and integrated with [Gymnasium](https://github.com/Farama-Foundation/Gymnasium).

---

## Supported Algorithms

- **PPO** – Proximal Policy Optimization
- **TD3** – Twin Delayed Deep Deterministic Policy Gradient

Each can be trained and evaluated in both BFS and cylinder configurations.

---

## Usage

# 1. Create a virtual environment
<pre><code>python3 -m venv rl_env</code></pre>

# 2. Activate the environment
<pre><code>source rl_env/bin/activate</code></pre>

# 3. Upgrade pip (recommended)
<pre><code>pip install --upgrade pip</code></pre>

# 4. Install dependencies
<pre><code>pip install -r requirements.txt</code></pre>

# 5. How to Run the Project.
 All the training and testing routines are launched through a unified entry point: `main.py`. This script dispatches the appropriate training or inspection script based on user-defined command-line arguments.
 <pre><code>python3 main.py --case {bfs|cylinder} --mode {train|test} --algo {ppo|td3} --control {amplitude|ampfreq|2jets|3jets [--submode {default|mu}] [--run-name "run_name"]</code></pre>
  
 Arguments: --case Geometry to simulate: bfs (Backward-Facing Step) or cylinder --mode Whether to train a new model or test an existing one --algo Reinforcement learning algorithm: ppo or td3 --control Control configuration: - For bfs: amplitude or ampfreq (amplitude + frequency control) - For cylinder: 2jets or 3jets --submode [Only for test] default rollout or mu (parametric viscosity test, bfs only) --run-name [Only for test] name of the folder in runs/ containing the trained model ---
   
 ##  Examples 🔹 Train PPO on BFS with amplitude + frequency control:<pre><code> python3 main.py --case bfs --mode train --algo ppo --control ampfreq </code></pre>🔹 Train TD3 on Cylinder with 2 jets: <pre><code>python3 main.py --case cylinder --mode train --algo td3 --control 2jets</code></pre> 🔹 Run rollout inspection (default) for PPO on BFS: <pre><code> python3 main.py --case bfs --mode test --algo ppo --control amplitude --submode default --run-name "run_name" </code></pre> 🔹 Run parametric test (μ sweep) for TD3 on BFS: <pre><code>python3 main.py --case bfs --mode test --algo td3 --control ampfreq --submode mu --run-name "run_name"</code></pre>


