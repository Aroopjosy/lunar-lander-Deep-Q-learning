import gymnasium as gym
import torch
from dqn import DeepQNetwork
env = gym.make("LunarLander-v3", render_mode="human")
# env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
            #    enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode="human")
observation, info = env.reset()

model = DeepQNetwork(0.001, [8], 256, 256, 4)

model.load_state_dict(torch.load('dqn_lunar_lander.pth', weights_only=True))
model.to(model.device)
model.eval()

for _ in range(10000):
    state = torch.from_numpy(observation).to(model.device)
    # print("state:", state)
    actions = model.forward(state)
    action = torch.argmax(actions).item()
    observation, reward, terminated, truncated, info = env.step(action)
   
    if terminated or truncated:
        observation, info = env.reset()
env.close()