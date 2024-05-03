from gym.envs.registration import register

register(
    id='PandaGraspingEnv-v0',
    entry_point='panda_gym.envs:PandaGraspingEnv',
    max_episode_steps=150
)
