# Original imports
import argparse
import logging
import time
import os
import shutil

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

def save(mixer, env, agent_type, results=None, episode="Final"): 
    # save agents
    # create folder for episode if it does not exist
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("results/episode_{0}".format(episode)):
        os.makedirs("results/episode_{0}".format(episode))
    if mixer is None:
        for i in range(len(env.players)):
            player = env.players[i]
            try:
                player.save("results/episode_{0}".format(episode), i)
            except Exception as e:
                print("Could not save agent:")
                print(e)
                pass
    else:
        mixer.save("results/episode_{0}".format(episode))

    if results is not None:
        # save results
        metrics.save_results(results[episode], episode=episode)
        if episode % 10000 == 0:
            # save plot every 10000 episodes
            metrics.compare_results(results, title="Partial plot",
                path="results/partial_plot.png".format(episode))
        

def _game_loop(env, render, mixer = None, episode=0, eval=False):
    # Reset env to start a new game episode
    nobs, ninfo = env.reset()
    steps = 0
    done = False

    # Render the game if requested by user with flag --render
    if render:
        env.render()
        time.sleep(0.5)

    # Episode loop
    # ends if all food items are collected or if max steps are reached
    while not done:
        steps += 1

        # Choose an action for each player
        actions = []
        if mixer is None:
            # If there is no mixer, each player will choose its own action
            for i in range(len(env.players)):
                player = env.players[i]
                action = player.choose_action(nobs[i]) if not eval else player.choose_optimal_action(nobs[i])
                actions.append(action)
        else:
            # If there is a mixer, it will return the joint action
            actions = mixer.choose_action(nobs) if not eval else mixer.choose_optimal_action(nobs)

        nobs, nreward, ndone, ninfo = env.step(actions)

        done = np.all(ndone)

        if eval:
            # If we are evaluating, we don't want to learn
            continue

        if mixer is None: 
            # If there is no mixer, each player will learn individually
            for i in range(len(env.players)):
                player = env.players[i]
                player.step(nobs[i], nreward[i], ndone[i], episode)
        else:
            # If there is a mixer, it will handle the learning of all agents
            mixer.step(nobs, nreward, ndone, ninfo, episode)

        if render:
            env.render()
            time.sleep(0.5)

    return steps, env


def main(game_count=1, render=False):
    # Set parameters for the environment
    s=8
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

    # erase results folder
    if os.path.exists("results"):
        for filename in os.listdir("results"):
            file_path = os.path.join("results", filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print("Could not delete file:")
                print(e)
                pass
        # erase folders as well
        for folder in os.listdir("results"):
            folder_path = os.path.join("results", folder)
            try:
                if os.path.isdir(folder_path):
                    shutil.rmtree(folder_path)
            except Exception as e:
                print("Could not delete folder:")
                print(e)
                pass

    # Define if we are using a mixer (QMIX) or not
    mixer = True
    # mixer = None
    
    # Define the type of agent
    # Current options are DQN or Random
    agent_type = "DQN"
    # agent_type = "Random"
        
    if mixer is None:
        # If we are not using a mixer, we need to define the agents
        # there must be one agent per player
        if agent_type == "DQN":
            agents = [DQNAgent for _ in range(len(env.players))]
        else:
            agents = [RandomAgent for _ in range(len(env.players))]
        for i in range(len(env.players)):
            player = env.players[i]
            player.set_controller(agents[i](env.players[i]))
            if player.name == "DQN Agent":
                player.init_from_env(env, game_count)
    else:
        # If we are using a mixer, we need to create the mixer
        # and add the players whose actions will be controlled
        # by the inner agent networks of the mixer
        mixer = QMIX_Controller(env.players[0])
        for i in range(1, len(env.players)):
            player = env.players[i]
            player.set_controller(mixer.add_player(env.players[i]))
        mixer.init_from_env(env, game_count)

    
    evaluation_duration = 25
    evaluation_frequency = game_count // 100
    if game_count < 1000:
        evaluation_frequency = game_count // 5
        evaluation_duration = 2

    # Then we run the game for the specified number of episodes
    # and collect the results in the episode_results dict
    episode_results = {}
    for episode in trange(game_count, position=0, desc="Training", leave=True):
        steps, env = _game_loop(env, render, mixer, episode)

        # evaluate every "evaluation_frequency" episodes
        if episode % evaluation_frequency == 0:
            all_steps = []
            all_player_scores = []
            all_total_score = []
            for _ in trange(evaluation_duration, position=1, desc="Evaluating in episode {}".format(episode), leave=False):
                steps, env = _game_loop(env, False, mixer, episode, True)
                player_scores = [player.score for player in env.players]
                # score should be 1 if all food is collected
                # and 0 if not
                all_steps.append(steps)
                all_player_scores.append(player_scores)
                all_total_score.append(sum(player_scores))

            episode_results[episode] = {
                "steps": np.mean(all_steps),
                "player_scores": [np.mean([player_scores[i] for player_scores in all_player_scores]) for i in range(len(env.players))],
                "score": np.mean(all_total_score),
                "min": np.min(all_total_score).astype('float64'),
                "max": np.max(all_total_score).astype('float64'),
            }
            save(mixer, env, agent_type, episode_results, episode)

    save(mixer, env, agent_type)
    
    # Finally, we call the compare_results function
    # to generate a final plot that will be saved on the /results folder
    metrics.save_results(episode_results)
    metrics.compare_results(episode_results, title="{0} on Foraging-10x10-3p-4f-v2".format("QMIX" if mixer is not None else agent_type),
                            with_additional_curves=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the level foraging game.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--times", type=int, default=1, help="How many times to run the game"
    )

    args = parser.parse_args()
    main(args.times, args.render)
