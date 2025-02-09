# HalfCheetah Reinforcement Learning

This project implements a Soft Actor-Critic (SAC) agent to train the HalfCheetah-v4 environment from the MuJoCo simulator. The agent learns to control the cheetah to run as fast as possible.

## Installation

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `mujoco_rl/src/`: Core implementation
  - `models.py`: Neural network architectures
  - `agent.py`: SAC agent implementation
- `mujoco_rl/train_cheetah.py`: Training script
- `mujoco_rl/test_env.py`: Testing script for trained models

## Training

To train the agent:

```bash
cd mujoco_rl
python train_cheetah.py
```

The script will:
- Train for 1000 episodes
- Save the best performing model as `cheetah_best_actor`
- Save the final model as `cheetah_final_actor`
- Generate training progress plots

## Testing

To test the trained models:

```bash
cd mujoco_rl
python test_env.py
```

This will run both the best and final models in the environment with rendering enabled.

## Features

- Soft Actor-Critic (SAC) implementation with automatic entropy tuning
- Experience replay buffer for off-policy learning
- Progress visualization during training
- Model checkpointing (best and final models)
- Configurable hyperparameters

## Results

The training progress can be monitored through:
- Terminal output showing episode rewards
- Progress plot saved as `cheetah_progress.png`
- Final results plot saved as `cheetah_rewards.png` 