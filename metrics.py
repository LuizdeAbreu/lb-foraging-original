import math
import numpy as np
import matplotlib.pyplot as plt

def z_table(confidence):
    return {
        0.90: 1.645,
        0.95: 1.96,
        0.98: 2.326,
        0.99: 2.576,
    }[confidence]

def std_error(std, n, confidence):
    return z_table(confidence) * std / math.sqrt(n)

def compare_results(results, confidence=0.95, title="Results"):
    figure = plt.figure()
    figure.suptitle(title)
    ax = figure.add_subplot(111)
    scores = np.array([result["score"] for _, result in results.items()])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.grid(True)
    ax.plot(scores, label="Score")
    ax.plot(
        [np.mean(scores)] * len(scores),
        label="Mean",
        linestyle="--",
        color="black",
    )
    plt.show()
