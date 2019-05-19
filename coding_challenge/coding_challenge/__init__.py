from gym.envs.registration import register

register(
    id='Battleship-v0',
    entry_point='coding_challenge.battleship:BattleshipEnv',
)

