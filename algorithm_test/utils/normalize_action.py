import numpy as np
import gym

class NormalizeActions(gym.ActionWrapper):
    
    def __init__(self, env):
        super(NormalizeActions, self).__init__(env)
        self.low_bound = self.action_space.low # -2
        self.upper_bound = self.action_space.high # +2
        
    def action(self, action):
        action = self.low_bound + (action + 1.0) * 0.5 * (self.upper_bound - self.low_bound)
        action = np.clip(action, self.low_bound, self.upper_bound)
        
        return action
    
    def reverse_action(self, action):
        action = 2 * (action - self.low_bound) / (self.upper_bound - self.low_bound) - 1
        action = np.clip(action, self.low_bound, self.upper_bound)
        
        return actions
