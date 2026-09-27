# ICMAPE
Multi-agent cooperative bandit algorithms for capacity identification and signal localization.
## Installation


```bash
pip install -r requirements.txt
```
---
## Capacity Identification 

Agents must collaboratively identify the highest-capacity arm of a multi-armed bandit within a fixed horizon. The environment is decentralized: each agent observes only its own rewards and must coordinate without direct communication.

### Environments

| Model | Config key | Reward distribution |
| ---  | --- | --- |
| Deterministic (Gaussian, std=0) | `+experiments/sequential/deterministic=...` | Fixed reward per arm |
| Stochastic (Bernoulli) | `+experiments/sequential/bernoulli=...` | Bernoulli reward per arm |

### Algorithms

| Algorithm | Config value | Description |
|---|---|---|
| Q-Learning (INDEP) | `MULTICPE_QLEARNING` | Independent per-agent Q-networks |
| Q-Learning (VDN) | `MULTICPE_QLEARNING` | Value decomposition: sum of per-agent Q-values |
| Uniform (baseline) | `MULTICPE_QLEARNING` | No learning; uniform random exploration only |
| TD3 | `MULTICPE_TD3` | Multi-agent TD3 with LSTM backbone and Gumbel-Softmax discrete actions |

---
## Training

All runs use `train_sequential.py` with [Hydra](https://hydra.cc) for configuration.
### Key arguments

| Argument | Description |
|---|---|
| `environment.horizon` | Episode length (number of arm pulls per episode) |
| `environment.num_actions` | Number of arms |
| `environment.num_players` | Number of agents |
| `training.max_training_steps` | Total environment steps |
| `seed` | Random seed |

---
### Q-Learning — Deterministic environment
**INDEP** (independent Q-networks, no value mixing):

```bash
python -u train_sequential.py \
+experiments/sequential/deterministic=multicpe_qlearning \
environment.horizon=<horizon> \
environment.num_actions=<num_arms> \
environment.num_players=<num_agents> \
training.max_training_steps=<steps> \
agent.critic_network.VDN_mode=False \
agent.critic_network.use_q_mixer=False \
agent.uniform_action_only=False \
experiment.seed=<seed>
```

**VDN** (value decomposition — sums per-agent Q-values):

```bash
python -u train_sequential.py \
+experiments/sequential/deterministic=multicpe_qlearning \
environment.horizon=<horizon> \
environment.num_actions=<num_arms> \
environment.num_players=<num_agents> \
training.max_training_steps=<steps> \
agent.critic_network.VDN_mode=True \
agent.critic_network.use_q_mixer=False \
agent.uniform_action_only=False \
experiment.seed=<seed>
```

**Uniform** (baseline — no RL training, purely random exploration):

```bash
python -u train_sequential.py \
+experiments/sequential/deterministic=multicpe_qlearning \
environment.horizon=<horizon> \
environment.num_actions=<num_arms> \
environment.num_players=<num_agents> \
training.max_training_steps=<steps> \
agent.uniform_action_only=True \
experiment.seed=<seed>
```


### TD3 — Deterministic environment
```bash
python -u train_sequential.py \
+experiments/sequential/deterministic=multicpe_td3_lstm \
environment.horizon=<horizon> \
environment.num_actions=<num_arms> \
environment.num_players=<num_agents> \
training.max_training_steps=<steps> \
experiment.seed=<seed>
```
---
### Q-Learning — Bernoulli environment
**INDEP**:

```bash
python -u train_sequential.py \
+experiments/sequential/bernoulli=multicpe_qlearning \
environment.horizon=<horizon> \
environment.num_actions=<num_arms> \
environment.num_players=<num_agents> \
training.max_training_steps=<steps> \
agent.critic_network.VDN_mode=False \
agent.critic_network.use_q_mixer=False \
agent.uniform_action_only=False \
experiment.seed=<seed>
```

**VDN**:

```bash
python -u train_sequential.py \
+experiments/sequential/bernoulli=multicpe_qlearning \
environment.horizon=<horizon> \
environment.num_actions=<num_arms> \
environment.num_players=<num_agents> \
training.max_training_steps=<steps> \
agent.critic_network.VDN_mode=True \
agent.critic_network.use_q_mixer=False \
agent.uniform_action_only=False \
experiment.seed=<seed>
```

**Uniform**:

```bash
python -u train_sequential.py \
+experiments/sequential/bernoulli=multicpe_qlearning \
environment.horizon=<horizon> \
environment.num_actions=<num_arms> \
environment.num_players=<num_agents> \
training.max_training_steps=<steps> \
agent.uniform_action_only=True \
experiment.seed=<seed>
```

  
#### TD3 — Bernoulli environment
```bash
python -u train_sequential.py \
+experiments/sequential/bernoulli=multicpe_td3_lstm \
environment.horizon=<horizon> \
environment.num_actions=<num_arms> \
environment.num_players=<num_agents> \
training.max_training_steps=<steps> \
experiment.seed=<seed>
```

---
## Example: 3-agent, 2-arm, horizon 20
```bash
# Q-Learning INDEP, bernoulli, 5000000 steps, seed 42

python -u train_sequential.py \
+experiments/sequential/bernoulli=multicpe_qlearning \
environment.horizon=20 \
environment.num_actions=2 \
environment.num_players=3 \
training.max_training_steps=500000 \
agent.critic_network.VDN_mode=False \
agent.critic_network.use_q_mixer=False \
agent.uniform_action_only=False \
experiment.seed=42
  

# TD3, bernoulli, 500000 steps, seed 42

python -u train_sequential.py \
+experiments/sequential/bernoulli=multicpe_td3_lstm \
environment.horizon=20 \
environment.num_actions=2 \
environment.num_players=3 \
training.max_training_steps=500000 \
experiment.seed=42
```

---
## Evaluation
All evaluation scripts live in `evaluation_scripts/bandit/`. Each script has a short configuration block at the top — edit those variables before running.

### Step 1 — Main evaluation (`result_bandit_capacity.py`)


Loads one or more saved `.pkl` model files, runs batched rollouts, and writes per-model results. The agent class (ICPE / MULTICPE\_TD3) is auto-detected from the model filename prefix (`icpe_*` or `MULTICPE_TD3_*`).

**Configure at top of script:**

| Variable | Description |
|---|---|
| `MODEL_PATHS` | List of paths to saved `.pkl` model files |
| `TOTAL_SAMPLES` | Number of evaluation episodes (default 10 000) |
| `BATCH_SIZE` | Episodes per batch — tune to fit VRAM (default 1 000) |
| `OUT_DIR` | Output directory, e.g. `results_summary/bandit/A5P3` |


```bash
python evaluation_scripts/bandit/result_bandit_capacity.py
```


**Outputs per model** (written to `OUT_DIR`):

- `results_<tag>.npz` — per-episode accuracy, stopping time, co-pull histogram

- `action_heatmap_<tag>.png` — heatmap of simultaneous arm-pull distribution

The `<tag>` encodes method, horizon, arms, players, and seed, e.g. `ICPE_IND_H20_K2_P3_seed42`. These `.npz` files feed into all plot scripts below.

### Step 2 — Prior robustness evaluation

#### Beta-distributed arm means (`evaluete_beta_bandit.py`)


Tests whether the agent generalises when arm means are drawn from $\text{Beta}(\alpha, \beta)$ instead of the uniform training prior. Sweeps all 25 combinations of $\alpha, \beta \in \{0.5, 1, 3, 5, 7\}$.
**Configure at top of script:** `MODEL_PATH` and `OUT_DIR`

```bash
python evaluation_scripts/bandit/evaluete_beta_bandit.py
```

**Outputs:**

- `results_beta_<tag>_a<alpha>_b<beta>.npz` — raw per-episode data for each $(\alpha, \beta)$ pair

- `summary_beta_<tag>.npz` — $5 \times 5$ accuracy and stop-step grids (used by the plot script)

  

#### Dirichlet-distributed arm capacities (`evaluete_dirichlet_bandit.py`)


Tests robustness when arm capacities are drawn from $\text{Dirichlet-Categorical}(\alpha)$. Sweeps concentration values $\alpha \in \{0.1, 0.5, 1, 5, 10\}$ and records per-episode total variation (TV) distance from the uniform prior alongside accuracy.

  
**Configure at top of script:** `MODEL_PATH` and `OUT_DIR`


```bash
python evaluation_scripts/bandit/evaluete_dirichlet_bandit.py
```

**Outputs:**

- `results_dirichlet_<tag>_a<alpha>.npz` — per-episode accuracy, inference probability, TV distance


---

## Plotting
### Accuracy and stopping-step bar charts (`plot_bandit_accuracy.py`)


Compares VDN, INDEP, TD3, and Uniform across seeds for each experimental condition. Reads all `results_*.npz` files from each condition subdirectory.


**Configure at top of script:** set `RESULT_DIR` (root results path) and `CONDITIONS` (dict mapping label → subdirectory name, e.g. `{"A3P5": "A3P5", "A5P3": "A5P3"}`).

```bash
python evaluation_scripts/bandit/plot_bandit_accuracy.py
```


**Outputs:** one grouped bar figure per condition, plus a combined figure.

### Survival function plot (`plot_survival_bandit.py`)

Plots $S(t) = P(\tau > t)$ — probability the agent has not yet stopped at step $t$ — for each method across seeds.

**Configure at top of script:** set `BASE` (root results directory) and `SETTINGS` (list of `(subdirectory, subtitle)` pairs).


```bash
python evaluation_scripts/bandit/plot_survival_bandit.py
```


**Output:** `<BASE>/survival_plot_bandit.png`
### Beta robustness plots (`plot_beta_robustness_bandit.py`)

Reads `summary_beta_<tag>.npz` and the per-combo `.npz` files. Produces:

- Accuracy heatmap over the $5 \times 5$ $(\alpha, \beta)$ grid, with the training-prior cell ($\text{Beta}(1,1) = \text{Uniform}$) outlined in red

- Stopping-step heatmap over the same grid

- $\mathrm{KL}(\text{Beta} \| \text{Uniform})$ vs accuracy scatter and $\mathrm{KL}(\text{Uniform} \| \text{Beta})$ vs accuracy scatter

  
**Configure at top of script:** set `DATA_DIR` and `TAG` (the `<tag>` string in the `.npz` filenames).


```bash
python evaluation_scripts/bandit/plot_beta_robustness_bandit.py
```


**Outputs** (written to `DATA_DIR`):

- `heatmap_accuracy_beta_<tag>_highlighted.png`

- `heatmap_stopstep_beta_<tag>_highlighted.png`

- `scatter_kl_vs_acc_<tag>.png`


### Dirichlet robustness plots (`plot_dirichlet_robustness.py`)


Per concentration value, produces:

- Calibration gap $|\hat{p}_\tau - (1 - \delta)|$ vs TV distance scatter

- Binned accuracy vs TV distance bar chart with Wilson 95% CI

**Configure at top of script:** set `DATA_DIR`, `TAG`, and `CONCENTRATION_VALUES`.

```bash
python evaluation_scripts/bandit/plot_dirichlet_robustness.py
```


**Outputs** (written to `DATA_DIR`): `tv_calib_<tag>_a<alpha>.png` and `tv_acc_<tag>_a<alpha>.png` for each $\alpha$.

### Dirichlet summary plot (`plot_dirichlet_summary_bandit.py`)

Bar chart of overall accuracy vs Dirichlet concentration $\alpha$ with 95% CI. Loads all `results_dirichlet_*.npz` files from a single directory.

**Configure at top of script:** set `DATA_DIR`.


```bash
python evaluation_scripts/bandit/plot_dirichlet_summary_bandit.py
```


**Output:** `DATA_DIR/summary_dirichlet_accuracy.png`

---

  
## Configuration files


Base agent configs live in `configs/agent/sequential/`. Experiment-level overrides are in `configs/experiments/sequential/{deterministic,bernoulli}/`. Hydra merges them at runtime — experiment configs take precedence over base agent configs.

---

# Signal Localization (Robot Environment)

Agents navigate a 2D grid to locate a hidden signal source. Observations are noisy local views; a distractor source may be present. Agents must collectively identify the true signal cell and stop within a fixed horizon.

## Environment
A single environment type is used: `RobotExploration2` (two-source signal detection on a $G \times G$ grid).
### Key environment parameters

| Parameter | Config key | Default | Description |
|---|---|---|---|
| Grid size | `environment.grid_size` | 25 | Side length of the square grid |
| Local view | `environment.view_side` | 5 | Side length of each agent's local observation window |
| Horizon | `environment.horizon` | 35 | Max steps per episode |
| Agents | `environment.num_players` | 3 | Number of agents |
| Signal hit prob | `environment.signal_hit_prob` | 0.9 | Probability of a positive detection when at the signal cell |
| Observation noise | `environment.smoke_obs_sigma` | 0.05 | Gaussian noise std on observations |
| Signal decay | `environment.smoke_decay` | 0.3 | Rate at which the signal field decays with distance |
| Min signal distance | `environment.min_signal_dist` | 10 | Min Manhattan distance between signal and distractor cells |
| Distractor intensity | `environment.distractor_intensity_min` / `_max` | 0.0 / 0.7 | Range of distractor intensity sampled per episode |

## Algorithms

| Algorithm | Config value | Description |
|---|---|---|
| Q-Learning (INDEP) | `ROBOT_MULTICPE_QLEARNING` | Independent per-agent Q-networks (Double DQN, epsilon-greedy) |
| Q-Learning (VDN) | `ROBOT_MULTICPE_QLEARNING` | Value decomposition: sum of per-agent Q-values |
| Uniform (baseline) | `ROBOT_MULTICPE_QLEARNING` | No learning; uniform random exploration only |
| SAC | `ROBOT_MULTICPE_SAC` | Soft actor-critic with CNN-LSTM backbone |
| TD3 | `ROBOT_MULTICPE_TD3` | TD3 with Gumbel straight-through actor and twin critics |

---
## Training

All runs use `train_robot.py` with [Hydra](https://hydra.cc) for configuration.

### Key arguments

| Argument | Description |
|---|---|
| `environment.horizon` | Episode length |
| `environment.num_players` | Number of agents |
| `environment.grid_size` | Grid side length |
| `environment.view_side` | Agent local view window side length |
| `environment.smoke_decay` | Signal decay rate |
| `environment.smoke_obs_sigma` | Observation noise std |
| `environment.min_signal_dist` | Min Manhattan distance between signal and distractor |
| `training.max_training_steps` | Total environment steps |
| `experiment.seed` | Random seed |

### Q-Learning


**INDEP** (independent Q-networks):

```bash
python -u train_robot.py \
+experiments/robot/SignalDetector2=robot_multicpe_qlearning \
environment.grid_size=<grid_size> \
environment.horizon=<horizon> \
environment.num_players=<num_agents> \
training.max_training_steps=<steps> \
agent.critic_network.VDN_mode=False \
agent.uniform_action_only=False \
experiment.seed=<seed>
```

**VDN** (value decomposition):

```bash
python -u train_robot.py \
+experiments/robot/SignalDetector2=robot_multicpe_qlearning \
environment.grid_size=<grid_size> \
environment.horizon=<horizon> \
environment.num_players=<num_agents> \
training.max_training_steps=<steps> \
agent.critic_network.VDN_mode=True \
agent.uniform_action_only=False \
experiment.seed=<seed>
```

**Uniform** (baseline):

```bash
python -u train_robot.py \
+experiments/robot/SignalDetector2=robot_multicpe_qlearning \
environment.grid_size=<grid_size> \
environment.horizon=<horizon> \
environment.num_players=<num_agents> \
training.max_training_steps=<steps> \
agent.uniform_action_only=True \
experiment.seed=<seed>
```



#### TD3

```bash
python -u train_robot.py \
+experiments/robot/SignalDetector2=robot_multicpe_td3 \
environment.grid_size=<grid_size> \
environment.horizon=<horizon> \
environment.num_players=<num_agents> \
training.max_training_steps=<steps> \
experiment.seed=<seed>
```

## Example: 3-agent, 25x25 grid, horizon 35

```bash
# TD3, 500 000 steps, seed 42

python -u train_robot.py \
+experiments/robot/SignalDetector2=robot_multicpe_sac \
environment.grid_size=25 \
environment.horizon=35 \
environment.num_players=3 \
training.max_training_steps=500000 \
experiment.seed=42
```

---

## Evaluation

All evaluation scripts live in `evaluation_scripts/signal/`. Each script has a configuration block at the top — edit those variables before running.
### Step 1 — Main evaluation (`result_robot_explore.py`)


Loads one or more saved `.pkl` model files and runs batched rollouts. Agent class is auto-detected from the filename prefix (`ROBOT_MULTICPE_SAC_*`, `ROBOT_MULTICPE_TD3_*`, `ROBOT_MULTICPE_QLEARNING_*`).

**Configure at top of script:**

| Variable | Description |
|---|---|
| `MODEL_PATHS` | List of paths to saved `.pkl` model files |
| `TOTAL_SAMPLES` | Number of evaluation episodes (default 10 000) |
| `BATCH_SIZE` | Episodes per batch — tune to fit VRAM (default 1 000) |
| `OUT_DIR` | Output directory, e.g. `result_summary/robot/explore` |


```bash
python evaluation_scripts/signal/result_robot_explore.py
```

**Output per model** (written to `OUT_DIR`):

- `results_<tag>.npz` — per-episode accuracy, stopping time, and metadata

The `<tag>` encodes method, horizon, grid size, view size, and seed, e.g. `TD3_H35_G25_V5_seed42`.

### Step 2 — Robustness evaluation (`evaluate_beta_robot_env.py`)

Tests whether the agent generalises when the distractor intensity is drawn from $\text{Beta}(\alpha, \beta)$ scaled to $[\text{intensity\_min}, \text{intensity\_max}]$ instead of the uniform training prior. Sweeps all 25 combinations of $\alpha, \beta \in \{0.5, 1, 3, 5, 7\}$.

**Configure at top of script:**`MODEL_PATHS`  and `OUT_DIR`

```bash
python evaluation_scripts/signal/evaluate_beta_robot_env.py
```

**Outputs** (per model, written to `OUT_DIR/<tag>/`):

- `results_beta_<tag>_a<alpha>_b<beta>.npz` — per-episode data for each $(\alpha, \beta)$ pair

- `summary_beta_<tag>.npz` — $5 \times 5$ accuracy and stop-step grids


---

## Plotting

### Accuracy and stopping-step bar charts (`plot_robot_accuracy.py`)

Compares VDN, INDEP, TD3, and Uniform across seeds. Reads all `results_*.npz` files from the results directory.


**Configure at top of script:** set `RESULT_DIR`.


```bash
python evaluation_scripts/signal/plot_robot_accuracy.py
```


**Output:** bar chart figures saved to `RESULT_DIR`.


### Survival function plot (`plot_survival_robot.py`)

Plots $S(t) = P(\tau > t)$ for each method across seeds. Reads method labels directly from the saved `.npz` files.

**Configure at top of script:** set `IN_DIR` (same as `OUT_DIR` from `result_robot_explore.py`).

```bash
python evaluation_scripts/signal/plot_survival_robot.py
```

**Output:** `IN_DIR/survival_plot.png`

### Accuracy vs signal-distractor distance (`plot_signal_distance_accuracy.py`)

Evaluates accuracy as a function of the Manhattan distance between the true signal and the distractor. For each target distance, the environment's valid pair pool is restricted to that distance, giving clean per-distance samples.

**Configure at top of script:** `MODEL_PATHS` and `OUT_DIR`
  
```bash
python evaluation_scripts/signal/plot_signal_distance_accuracy.py
```

**Outputs per model** (written to `OUT_DIR`):

- `distance_accuracy_<tag>.npz` — raw accuracy per distance
- `distance_accuracy_<tag>.png` — bar chart of accuracy vs distance

### Beta robustness plots (`plot_beta_robustness_robot.py`)

Reads `summary_beta_<tag>.npz` and the per-combo raw files. Produces the same set of plots as the bandit version:

- Accuracy heatmap over the $5 \times 5$ $(\alpha, \beta)$ grid, with the training-prior cell ($\text{Beta}(1,1) = \text{Uniform}$) outlined in red

- Stopping-step heatmap

- $\mathrm{KL}(\text{Beta} \| \text{Uniform})$ vs accuracy scatter

**Configure at top of script:** set `BASE_DIR` (root containing per-tag subdirectories). Set `SUMMARY_PATHS` to a list of specific files, or leave as `None` to auto-discover.

```bash
python evaluation_scripts/signal/plot_beta_robustness_robot.py
```


**Outputs** (written alongside each `summary_beta_<tag>.npz`):

- `heatmap_accuracy_beta_<tag>_highlighted.png`
- `heatmap_stopstep_beta_<tag>_highlighted.png`
- `scatter_kl_vs_acc_<tag>.png`


---

## Configuration files

Robot base agent configs live in `configs/agent/robot/`. Experiment-level overrides are in `configs/experiments/robot/SignalDetector2/`. The root config is `configs/config_robot.yaml`.