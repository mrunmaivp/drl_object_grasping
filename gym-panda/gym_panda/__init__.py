from gymnasium.envs.registration import register

register(
    id='PandaGraspingEnv-v0',
    entry_point='gym_panda.envs:PandaGraspingEnv',
    max_episode_steps=100
)

register(
    id='GraspingEnv-v0',
    entry_point='gym_panda.envs:GraspingEnv',
    max_episode_steps=100
)