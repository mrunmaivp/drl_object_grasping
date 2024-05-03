#! /usr/bin/env python

import gym

env = gym.make('PandaGrasping-v0')

env = DummyVecEnv([lambda: env])
model.save("sac_grasping_model")

loaded_model = SAC.load("sac_grasping_model")
model = SAC("MlpPolicy", env, verbose=1)

# Train the agent
total_timesteps = 100000  # Adjust as needed
model.learn(total_timesteps=total_timesteps)

obs = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()

env.close()