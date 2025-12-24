from collections import deque
import random
import numpy as np
import pandas as pd
import csv

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def save_to_csv(self, filename):
        """Save the contents of memory to a CSV file."""
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(['state', 'action', 'reward', 'next_state', 'done'])
            # Write the data
            for entry in self.memory:
                writer.writerow(entry)

    def to_dataset(self):
        """Convert memory to a pandas DataFrame."""
        return pd.DataFrame(self.memory, columns=['state', 'action', 'reward', 'next_state','done'])

class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, state, action, reward, next_state, done):

        state      = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):

        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done