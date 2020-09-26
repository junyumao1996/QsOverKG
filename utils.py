import numpy as np
import os 
from pathlib import Path
import matplotlib.pyplot as plt
import copy


def win_logging(win_log, win_curve, exp_dir, i, N_GAMES, N_AVERAGE, T):
    success_rate = np.average(np.array(win_log))
    print("Game {}/{} - Success rate: {}".format(i + 1, N_GAMES, success_rate))
    # exit()
    win_curve_copy = copy.copy(win_curve)
    win_curve_copy.append(success_rate)
    fig = plt.figure()
    plt.plot(N_AVERAGE*np.arange(len(win_curve_copy))*T, win_curve_copy)
    plt.xlabel("Game Steps")
    plt.ylabel("Winning Rate")
    plt.savefig(os.path.join(exp_dir, 'win_curve.png'))
    plt.close(fig)
    win_log_l = [str(i) for i in win_log]
    string = ' '.join(win_log_l)
    with open(os.path.join(exp_dir, 'raw.txt'), "a") as myfile:
        myfile.write(string)
    np.save(os.path.join(exp_dir, 'win_rate.npy'), win_curve_copy)
    return success_rate, win_curve_copy

def metric_logging(win_log, win_curve, exp_dir, i, N_GAMES, N_AVERAGE, T, kbr_curve):
    success_rate = np.average(np.array(win_log))
    win_curve_copy = copy.copy(win_curve)
    win_curve_copy.append(success_rate)
    fig = plt.figure()
    plt.plot(N_AVERAGE*np.arange(len(win_curve_copy))*T, win_curve_copy)
    plt.xlabel("Game Steps")
    plt.ylabel("Winning Rate")
    plt.savefig(os.path.join(exp_dir, 'win_curve.png'))
    plt.close(fig)
    win_log_l = [str(i) for i in win_log]
    string = ' '.join(win_log_l)
    with open(os.path.join(exp_dir, 'raw.txt'), "a") as myfile:
        myfile.write(string)
    np.save(os.path.join(exp_dir, 'win_rate.npy'), win_curve_copy)
    np.save(os.path.join(exp_dir, 'kbc_ratio.npy'), kbr_curve)
    print("Game {}/{} - Success rate: {} KBC ratio: {}".format(i + 1, N_GAMES, success_rate, kbr_curve[-1]))
    return success_rate, win_curve_copy

def KB_completion_ratio(ext_KB, int_KB):
    """
    Compute KB completion ratio of internal KB to external one.
    ext_KB, int_KB: two numpy array
    """
    e_size = len(ext_KB)
    i = 0
    for fact in int_KB:
        if fact in ext_KB:
            i += 1
    return i/e_size

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
    return accu_results 

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