import numpy as np
import matplotlib.pyplot as plt
from Ipython.display import clear_output

def smooth_plot(factor, item, plot_decay):
    item_x = np.arange(len(item))
    item_smooth = [np.mean(item[i:i+factor]) if i > factor else np.mean(item[0:i+1])
                  for i in range(len(item))]
    for i in range(len(item)// plot_decay):
        item_x = item_x[::2]
        item_smooth = item_smooth[::2]
    return item_x, item_smooth

def plot(episode, rewards, value_losses, policy_losses, noise):
    clear_output(True)
    rewards_x, rewards_smooth = smooth_plot(10, rewards, 500)
    value_losses_x, value_losses_smooth = smooth_plot(10, value_losses, 10000)
    policy_losses_x, policy_losses_smooth = smooth_plot(10, policy_losses, 10000)
    noise_x, noise_smooth = smooth_plot(10, noise, 100)

    plt.figure(figsize=(18, 12))
    plt.subplot(411)
    plt.title('episode %s. reward: %s'%(episode, rewards_smooth[-1]))
    plt.plot(rewards, label="Rewards", color='lightsteelblue', linewidth='1')
    plt.plot(rewards_x, rewards_smooth, label='Smothed_Rewards', color='darkorange', linewidth='3')
    plt.legend(loc='best')

    plt.subplot(412)
    plt.title('Value_Losses')
    plt.plot(value_losses,label="Value_Losses",color='lightsteelblue',linewidth='1')
    plt.plot(value_losses_x, value_losses_smooth,
             label="Smoothed_Value_Losses",color='darkorange',linewidth='3')
    plt.legend(loc='best')

    plt.subplot(413)
    plt.title('Policy_Losses')
    plt.plot(policy_losses,label="Policy_Losses",color='lightsteelblue',linewidth='1')
    plt.plot(policy_losses_x, policy_losses_smooth,
             label="Smoothed_Policy_Losses",color='darkorange',linewidth='3')
    plt.legend(loc='best')

    plt.subplot(414)
    plt.title('Noise')
    plt.plot(noise,label="Noise",color='lightsteelblue',linewidth='1')
    plt.plot(noise_x, noise_smooth,
             label="Smoothed_Noise",color='darkorange',linewidth='3')
    plt.legend(loc='best')

    plt.show()
