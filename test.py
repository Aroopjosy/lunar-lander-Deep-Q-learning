import gymnasium as gym
import torch
env = gym.make("LunarLander-v3", render_mode="human")
# env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
            #    enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode="human")
observation, info = env.reset(seed=42)

for _ in range(100):
    torch.load('dqn_lunar_lander.pth')

    observation, reward, terminated, truncated, info = env.step(action)
    print('Action:', action)
    print('Observation:', observation)
    print('Reward:', reward)
    print('Terminated:', terminated)
    print('Truncated:', truncated)
    print('Info:', info)


   
    if terminated or truncated:
        observation, info = env.reset()
env.close()