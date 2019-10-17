import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ValueNetwork, self).__init__()
        self.linear_state = nn.Linear(state_dim, 64)
        self.linear_action = nn.Linear(action_dim, 64)
        self.linear2 = nn.Linear(128, 32)
        self.linear3 = nn.Linear(32, 1)
    
    def forward(self, state, action):
        hidden_state = F.relu(self.linear_state(state))
        hidden_action = F.relu(self.linear_action(action))
        cat_state_action = torch.cat((hidden_action, hidden_state),dim=1)
        hidden2 = F.relu(self.linear2(cat_state_action))
        Q = self.linear3(hidden2)
        return Q

class PolicyNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PolicyNetwork, self).__init__()
        self.linear1 = nn.Linear(in_dim, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, out_dim) # (256, 1)
    
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x
    

class DDPG:
    def __init__(self, in_dim, out_dim, replay_buffer, device, learning_rate=1e-3, gamma=0.99, soft_tau=1e-2):
        self.replay_buffer = replay_buffer
        self.device = device
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        
        self.policy_net = PolicyNetwork(self.in_dim, self.out_dim).to(self.device)
        self.value_net = ValueNetwork(self.in_dim, self.out_dim).to(self.device)
        self.target_value_net = copy.deepcopy(self.value_net)
        self.target_policy_net = copy.deepcopy(self.policy_net)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.soft_tau = soft_tau
        
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        return self.policy_net(state).detach().cpu().numpy()
        
    def train(self):
        samples = self.replay_buffer.sample()
        state, action, next_state = samples['current_state'], samples['action'], samples['next_state']
        reward, done = samples['reward'], samples['done']
        
        with torch.no_grad():
            next_action = self.target_policy_net(next_state)
            next_action_value = self.target_value_net(next_state, next_action)
            target_value = reward + (1.0-done)*self.gamma*next_action_value
            
        value = self.value_net(state, action)
        
        value_loss = F.mse_loss(value, target_value)

        policy_loss = -self.value_net(state, self.policy_net(state)).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(target_param.data*(1.0-self.soft_tau) + param.data*self.soft_tau)
        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(target_param.data*(1.0-self.soft_tau) + param.data*self.soft_tau)

        return value_loss.item(), policy_loss.item()
    
    def store(self, state, action, next_state, reward, done):
        self.replay_buffer.store(state, action, next_state.flatten(), reward, done)
        
    def buffer_size(self):
        return len(self.replay_buffer)
    
    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename + "_policy.pth")
        print("actor model {} save succeed!".format(filename))
    
    def load(self, filename):
        self.policy_net.load_state_dict(torch.load(filename + "_policy.pth"))
        print("actor model {} load succeed!".format(filename))