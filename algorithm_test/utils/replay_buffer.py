import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, column, batch_size, buffer_size, device):
        self.current_state = np.zeros((buffer_size, column), dtype=np.float32)
        self.next_state = np.zeros((buffer_size, column), dtype=np.float32)
        init = lambda buffer_size : np.zeros(buffer_size, dtype=np.float32)
        self.action = init(buffer_size)
        self.reward = init(buffer_size)
        self.done = init(buffer_size)
        self.buffer_size, self.batch_size = buffer_size, batch_size
        self.size, self.current_index = 0, 0
        
        self.device = device
    
    def store(self, current_state, action, next_state, reward, done):
        self.current_state[self.current_index] = current_state
        self.action[self.current_index] = action
        self.next_state[self.current_index] = next_state
        self.reward[self.current_index] = reward
        self.done[self.current_index] = done
        self.current_index = (self.current_index + 1) % self.buffer_size
        self.size = min(self.buffer_size, (self.size + 1))
    
    def sample(self):
        index = np.random.choice(self.size, self.batch_size, replace=False)
        return dict(current_state = torch.tensor(self.current_state[index],dtype=torch.float).to(self.device),
                    action = torch.tensor(self.action[index]).reshape(-1, 1).to(self.device),
                    next_state = torch.tensor(self.next_state[index],dtype=torch.float).to(self.device),
                    reward = torch.tensor(self.reward[index]).reshape(-1, 1).to(self.device),
                    done = torch.tensor(self.done[index]).reshape(-1, 1).to(self.device))
    
    def __len__(self):
        return self.size
