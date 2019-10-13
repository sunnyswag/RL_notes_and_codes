import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import time
from datetime import datetime
import os

import sys
sys.path.append("../utils/")
from normalize_action import NormalizeActions
from replay_buffer import ReplayBuffer
from DDPG import DDPG

def train_ddpg(ddpg, writer, current_model_dir):
    test = True

    for episode in range(train_episodes):
        
        episode_start_time = datetime.now()
        
        state = env.reset()
        episode_reward = 0
        for step in range(train_steps):
            action1 = ddpg.get_action(state)
    
            if step % update_step == 0:
                test = not test
            noise_sample = abs(np.random.randn(1)) * noise_discount
            noise = noise_sample if test else -noise_sample
            action = action1 + noise

            next_state, reward, done, _ = env.step(action.flatten())

            ddpg.store(state, action, next_state.flatten(), reward, done)
            if ddpg.buffer_size() > batch_size :
                value_loss, policy_loss = ddpg.train()

            state = next_state
            episode_reward += reward
        
            if done:
                break
                
        episode_end_time = datetime.now()
        episode_delta_T = (train_end_time - train_start_time).seconds
        
        writer.add_scalars("train_reward/update_step_{}".format(update_step),
                           {"noise_discount_{}".format(noise_discount):episode_reward}, episode)
        print("Episode {}/{} , episode_reward {}, value_loss {}, policy_loss {}, {} seconds"
              .format(episode + 1, train_episodes, episode_reward, value_loss, policy_loss, episode_delta_T))
        
    ddpg.save(current_model_dir)

def test_ddpg(ddpg, writer, current_model_dir):
    ddpg.load(current_model_dir)
    for test_episode in range(test_episodes):
        state = env.reset()
        rewards = 0
        for _ in range(test_steps):
            action = ddpg.get_action(state.flatten())
            next_state, reward, done, info = env.step(action)
            state = next_state
            rewards += reward
            if done: break
        writer.add_scalars("test_reward/update_step_{}".format(update_step),
                           {"noise_discount_{}".format(noise_discount):rewards}, test_episode)
        print("Episode {}/{} , episode_reward {}"
              .format(test_episode + 1, test_episodes, episode_reward))

if __name__ == "__main__":
    
    ## hyperparameter
    
    env_name = "MountainCarContinuous-v0"

    current_time = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
    ROOT_DIR = "../test_log/DDPG_for_mountian_car/noise_and_step_{}".format(current_time)
    model_dir = os.path.join(ROOT_DIR, "model")
    plot_dir = os.path.join(ROOT_DIR, "tensorboard")
    
    print("model_dir : {}".format(model_dir))
    print("plot_dir : {}".format(plot_dir))
    
    os.makedirs(model_dir)
    os.makedirs(plot_dir)
    
    print("make model_dir and plot_dir succeed!")
    
    buffer_size = 1000000
    batch_size = 128
    learning_rate = 1e-3

    train_episodes = 200
    train_steps = 1000
    test_episodes = 100
    test_steps = 100

    update_steps = [50, 100, 150, 200, 250, 300]
    noise_discounts = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]

    env = NormalizeActions(gym.make(env_name))
    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.shape[0]
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    replay_buffer = ReplayBuffer(in_dim, batch_size, buffer_size, device)
    writer = SummaryWriter(plot_dir)
    
    ddpg = DDPG(in_dim, out_dim, replay_buffer, device, learning_rate)
    
    print("DDPG network init finish")
    
    ## hyperparameter
    print("")
    print("========== START TRAIN AND TEST ==========")
    print("")
    print("test update steps : {}".format(update_steps))
    print("test noise discounts : {}".format(noise_discounts))
    print("")
    
    for update_step in update_steps:
        for noise_discount in noise_discounts:
            current_model_dir = os.path.join(model_dir, "update_step_{}_noise_discount_{}"
                                             .format(update_step, noise_discount))
            # train
            
            print("=== START Train < update_step : {}, noise_discount : {} > ==="
                 .format(update_step, noise_discount)
            train_start_time = datetime.now()
            train_ddpg(ddpg, writer, current_model_dir)
            train_end_time = datetime.now()
            
            train_delta_T = round((train_end_time - train_start_time).seconds/60)
            print("=== END Train < update_step : {}, noise_discount : {} > {} minutes ==="
                  .format(update_step, noise_discount, train_delta_T))

            # test
            
            print("=== START Test < update_step : {}, noise_discount : {} > ==="
                 .format(update_step, noise_discount)
            test_start_time = datetime.now()
            test_ddpg(ddpg, writer, current_model_dir)
            test_end_time = datetime.now()
            
            test_delta_T = round((test_end_time - test_start_time).seconds/60)
            print("=== END Test  < update_step : {}, noise_discount : {} > {} minutes ==="
                  .format(update_step, noise_discount, test_delta_T))