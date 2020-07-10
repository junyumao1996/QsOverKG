import numpy as np
import os 
from pathlib import Path
import matplotlib.pyplot as plt


def winning_rate_appro(results, n=1000):
    """
    Compute moving average of the winning rate.
    :param n: number of averaged consequent games
    """
    # average over n games
    accu_results = []
    for i, r in enumerate(results):
        if i < n:
            accu_results.append(np.sum(np.array(results[:i+1]))/(i+1))
        else:
            accu_results.append(np.sum(np.array(results[i-n:i]))/n)
    return smoothed_results 

def winning_rate_smoothed(results, alpha, path, n=20):
    """
    Compute moving average of the winning rate.
    :param results: result sequence (list) of played games 
    :param alpha: moving average factor
    :param n: number of averaged consequent games
    """
    # average over n games
    accu_results = []
    for i, r in enumerate(results):
        if i < n:
            accu_results.append(np.sum(np.array(results[:i+1]))/(i+1))
        else:
            accu_results.append(np.sum(np.array(results[i-n:i]))/n)
    # moving average
    smoothed_results = []
    for i, r in enumerate(accu_results):
        if i == 0:
            smoothed_results.append(r)
        else:
            smoothed = r * alpha + (1 - alpha) * smoothed_results[-1]
            smoothed_results.append(smoothed)
    # save win curve
    fig = plt.figure()
    plt.plot(smoothed_results)
    plt.xlabel("Game Steps")
    plt.ylabel("Winning Rate")
    plt.savefig(os.path.join(path, 'win_curve.png'))
    return smoothed_results 