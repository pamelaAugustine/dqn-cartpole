import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from collections import deque
import random
import matplotlib.pyplot as plt

class DQNAgent:
    def __init__(self, state_size, action_size):
        """ Initialize the DQN agent. """
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters (Optimized)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995  # Faster decay to switch to exploitation
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.train_start = 5000  # More experience before training
        self.memory = deque(maxlen=50000)  # Larger buffer for better learning

        # Build main and target networks
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        """ Create a stronger NN for better learning. """
        model = tf.keras.Sequential()
        model.add(layers.Input(shape=(self.state_size,)))  
        model.add(layers.Dense(128, activation='relu'))  # More neurons
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """ Copy weights from main model to target model. """
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done, time_step):
        """ Store experience in replay buffer with better rewards. """
        if done:
            reward = -10  # Penalize falling early
        else:
            reward = reward + (time_step / 10)  # Reward lasting longer

        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """ Epsilon-greedy action selection. """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def train_replay(self):
        """ Train with mini-batch samples. """
        if len(self.memory) < self.train_start:
            return

        mini_batch = random.sample(self.memory, self.batch_size)
        states = np.array([exp[0][0] for exp in mini_batch])
        next_states = np.array([exp[3][0] for exp in mini_batch])

        q_values = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        for i, (state, action, reward, next_state, done) in enumerate(mini_batch):
            if done:
                q_values[i][action] = reward
            else:
                q_values[i][action] = reward + self.gamma * np.amax(next_q_values[i])

        self.model.fit(states, q_values, batch_size=self.batch_size, verbose=0)

        # Ensure epsilon decays properly
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

def train_dqn(episodes=500, update_target_freq=10):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    episode_rewards = []

    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])  
        done = False
        time_step = 0
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  

            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done, time_step)

            state = next_state
            time_step += 1
            total_reward += reward

            # Train every 2 steps
            if time_step % 2 == 0:
                agent.train_replay()

        if e % update_target_freq == 0:
            agent.update_target_model()

        # Print progress every 25 episodes
        if e % 25 == 0 or e == episodes - 1:
            print(f"Episode: {e}, Score: {time_step}, Epsilon: {agent.epsilon:.3f}")

        episode_rewards.append(total_reward)

    # Plot the rewards over episodes
    plt.plot(range(episodes), episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Performance on CartPole-v1')
    plt.show()
    env.close()

if __name__ == "__main__":
    train_dqn()

