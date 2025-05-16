# 🎯 Deep Q-Network (DQN) on CartPole

This project implements a Deep Q-Network agent to solve the classic `CartPole-v1` reinforcement learning environment using TensorFlow and `gymnasium`. It includes key DQN components like a replay buffer, target network, epsilon-greedy policy, and reward shaping.

> Developed as part of FLGF24 Week 22 RL Workshop.

---

## 🧠 Project Goal

Train an agent to learn a policy that keeps the pole balanced for as long as possible, achieving an average return of 200+ over consecutive episodes.

---

## ⚙️ Features Implemented

- ✅ Deep neural network with 2 hidden layers (128 units each)
- ✅ Epsilon-greedy exploration with decay
- ✅ Experience replay buffer (50,000 capacity)
- ✅ Target network updates every 10 episodes
- ✅ Reward shaping to encourage longer survival
- ✅ Custom training loop with episode tracking

---

## 🧪 Sample Results

| Metric            | Value         |
|-------------------|---------------|
| Max Episode Score | ~400 steps    |
| Training Episodes | 500           |
| Avg score @ 300+  | 200+ (stable) |
| Model Stability   | Moderate (room for tuning) |

---

## 📊 Learning Curve

At episode 300+, the model began performing consistently above 200 steps, though some episodes still showed variability. While no plot is currently included, logging results across episodes provided useful insights for future tuning.

> *(Plotting reward curves is a potential future improvement to visualize learning progress.)*


---

## 🧬 Key Learnings & Challenges

- Increased hidden layer size significantly improved learning stability.
- Replay buffer size had a direct impact on memory retention and performance.
- Performance fluctuated late in training — likely due to aggressive epsilon decay and sensitivity to state transitions.
- Reward shaping helped incentivize balance duration early on.

---

## 🧠 Architecture Summary

```python
model = tf.keras.Sequential([
    layers.Input(shape=(state_size,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(action_size, activation='linear')
])
```

---

## 📦 Tech Stack

- Python
- `gymnasium`
- TensorFlow / Keras
- `matplotlib` (reward curve)

---

## 🚀 Getting Started

```bash
pip install gymnasium tensorflow matplotlib
python dqn_cartpole.py
```

---

## 🗂️ File Structure

```
├── dqn_cartpole.py           # Main training script
├── README.md
```

---

## 👩‍💻 Author

Pamela Augustine  

