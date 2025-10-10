import numpy as np

class QLearningOptimizer:
    def __init__(self):
        self.actions = ['grid', 'solar', 'mix']
        self.q_table = np.zeros((3, 3))   # [state, action]
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.training_episodes = 0

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions)
        return self.actions[np.argmax(self.q_table[state])]

    def learn(self, state, action_idx, reward, next_state):
        predict = self.q_table[state, action_idx]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action_idx] += self.alpha * (target - predict)

    def simulate_episode(self):
        total_reward = 0
        for hour in range(24):
            state = hour % 3  # 0: morning, 1: noon, 2: evening
            action = self.choose_action(state)
            action_idx = self.actions.index(action)

            # Reward function: solar = higher reward
            if action == 'solar':
                reward = 10 - np.random.uniform(0, 1) * 2
            elif action == 'mix':
                reward = 7
            else:
                reward = 4 - np.random.uniform(0, 1)
            next_state = (hour + 1) % 3
            self.learn(state, action_idx, reward, next_state)
            total_reward += reward
        return total_reward

    def train(self, episodes=100):
        for _ in range(episodes):
            self.simulate_episode()
        self.training_episodes += episodes
        return self.q_table

    def optimize(self, solar_avail, demand):
        """Predict distribution after training."""
        if demand <= 0:
            return {'solar_used': 0, 'grid_used': 0, 'renewable_ratio_percent': 0}
        ratio = min(1.0, solar_avail / demand)
        pref = np.mean(self.q_table, axis=0)
        if pref[1] > pref[0]:  # prefer solar
            ratio *= 1.1
        ratio = min(ratio, 1.0)
        solar_used = demand * ratio
        grid_used = demand - solar_used
        return {
            'solar_used': round(solar_used, 2),
            'grid_used': round(grid_used, 2),
            'renewable_ratio_percent': round(ratio * 100, 2)
        }
