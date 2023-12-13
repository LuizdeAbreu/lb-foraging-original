import math
import numpy as np
import matplotlib.pyplot as plt
import json
import os

def z_table(confidence):
    return {
        0.90: 1.645,
        0.95: 1.96,
        0.98: 2.326,
        0.99: 2.576,
    }[confidence]

def std_error(std, n, confidence):
    return z_table(confidence) * std / math.sqrt(n)

def exponential_moving_average(data):
    n = len(data)
    weights = np.exp(np.linspace(-1., 0., n))
    weights /= weights.sum()
    ema = np.convolve(data, weights, mode='full')[:len(data)]
    return ema

def last_x_average(data, x):
    n = len(data)
    averages = np.zeros(n)
    for i in range(n):
        start_index = max(0, i - x - 1)  # Starting index for the last 10 values
        last_x = data[start_index:i+1]  # Extract the last 10 values up to index i
        averages[i] = np.mean(last_x)  # Calculate the mean and store it in the averages array
    return averages

def compare_results(results, confidence=0.95, title="Results"):
    # print("episode results", results)
    num_episodes = len(results)
    title = title + " (n={0})".format(num_episodes)
    figure = plt.figure()
    figure.suptitle(title)
    ax = figure.add_subplot(111)
    scores = np.array([result["score"] for _, result in results.items()])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.grid(True)
    x_axis_values = list(results.keys())
    # plot scores with x axis as episode number
    ax.plot(x_axis_values, scores, label="Score", linewidth=2)
    # xticks should only be the episode numbers where we evaluated the score
    ax.set_xticks(list(results.keys()))
    mean = np.mean(scores)
    # plot mean
    ax.axhline(mean, 
        label="Mean",
        linestyle="--",
        color="black",
    )
    

    ema_final_rewards = exponential_moving_average(scores)
    x = len(scores) // 10
    lastx_final_rewards = last_x_average(scores, x)
    ax.plot(x_axis_values, ema_final_rewards, label="Exponential moving average", linewidth=0.8)
    ax.plot(x_axis_values, lastx_final_rewards, label=f'Average of last {x} values', linewidth=0.8)
    plt.legend(loc="upper left")

    # save to results folder as png
    figure.savefig("results/{0}.png".format(title))
    
    plt.show()

def save_results(result, episode = None):
    # create eval results folder if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("results/evaluation"):
        os.makedirs("results/evaluation")

    path = "results/evaluation/"
    filename = "evaluation_results.json" if episode is None else "evaluation_results_{0}.json".format(episode)
    # save to results folder as json
    with open(path + filename, "w") as f:
        json.dump(result, f)