# Original imports
import argparse
import logging
import time
import gym
import numpy as np
from gym.envs.registration import register
from lbforaging.agents.random_agent import RandomAgent
# from lbforaging.agents.q_agent import QAgent
from lbforaging.agents.dqn_agent import DQNAgent
from enum import Enum
from tqdm import trange

from lbforaging.agents.qmix_controller import QMIX_Controller

class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5

# Original imports that were not used
# (keeping them here for safety)
import random
import lbforaging

# New imports
import metrics

logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym") 

def _game_loop(env, render, mixer = None, episode=0):
    """
    """
    nobs, ninfo = env.reset()
    steps = 0
    done = False

    if render:
        env.render()
        time.sleep(0.5)

    while not done:
        steps += 1

        actions = []
        if mixer is None:
            # each player will return an action
            for i in range(len(env.players)):
                player = env.players[i]
                actions.append(player.choose_action(nobs[i]))
        else:
            actions = mixer.choose_action(nobs)

        nobs, nreward, ndone, ninfo = env.step(actions)

        if mixer is None: 
            # each player will learn from the experience
            for i in range(len(env.players)):
                player = env.players[i]
                player.step(nobs[i], nreward[i], ndone[i], episode)
        else:
            mixer.step(nobs, nreward, ndone, ninfo, episode)

        if render:
            env.render()
            time.sleep(0.5)

        done = np.all(ndone)

    return steps, env


def main(game_count=1, render=False):
    s=6
    p=3
    f=4
    c=0
    env_id = "Foraging-{0}x{0}-{1}p-{2}f{3}-v2".format(s, p, f, "-coop" if c else "")
    register(
        id=env_id,
        entry_point="lbforaging.foraging:ForagingEnv",
        kwargs={
            "players": p,
            "max_player_level": 5,
            "field_size": (s, s),
            "max_food": f,
            "sight": s,
            "max_episode_steps": 50,
            "force_coop": c,
        },
    )
    env = gym.make(env_id)

    mixer = True
    # mixer = None
    
    agents = [DQNAgent for _ in range(len(env.players))]
    if mixer is None:
        for i in range(len(env.players)):
            player = env.players[i]
            player.set_controller(agents[i](env.players[i]))
            if player.name == "DQN Agent":
                player.init_from_env(env)
    else:
        mixer = QMIX_Controller(env.players[0])
        for i in range(1, len(env.players)):
            player = env.players[i]
            player.set_controller(mixer.add_player(env.players[i]))
        mixer.init_from_env(env)

    episode_results = {}
    for episode in trange(game_count):
        steps, env = _game_loop(env, render, mixer, episode)
        # create dict with results
        player_scores = [player.score for player in env.players]
        # score should be 1 if all food is collected
        episode_results[episode] = {
            "steps": steps,
            "player_scores": player_scores,
            "score": sum(player_scores),
        }
    
    # print("episode_results", episode_results)
    # compare results
    metrics.compare_results(episode_results, title="Foraging-10x10-3p-4f-v2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the level foraging game.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--times", type=int, default=1, help="How many times to run the game"
    )

    args = parser.parse_args()
    main(args.times, args.render)
