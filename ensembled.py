import pickle
from scipy.stats import spearmanr
import numpy as np

def ensemble_pred_scores_and_calculate_kendall_tau(filenames, weights=None):
    # Initialize a list to store all pred_scores arrays and one to store true_scores for validation
    all_pred_scores = []
    true_scores_reference = None
    
    # Load and validate true_scores are the same across all files, collect all pred_scores
    for filename in filenames:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            true_scores, pred_scores = zip(*data)
            true_scores = np.array(true_scores)
            pred_scores = np.array(pred_scores)
            if true_scores_reference is None:
                true_scores_reference = true_scores
            all_pred_scores.append(pred_scores)
            file_kendall_tau = spearmanr(true_scores, pred_scores).correlation
            print(f"SpearmanR for {filename}: {file_kendall_tau}")
    all_pred_scores = np.vstack(all_pred_scores)
    if weights:
        if len(weights) != len(all_pred_scores):
            raise ValueError("Number of weights must match the number of pred_scores arrays.")
        ensembled_pred_scores = np.average(all_pred_scores, axis=0, weights=weights)
    else:
        ensembled_pred_scores = np.mean(all_pred_scores, axis=0)
    
    # Calculate and return the SpearmanR correlation
    return spearmanr(true_scores_reference, ensembled_pred_scores).correlation


filenames = ["nb201_sorted_results_adj_gin_774.pkl", "nb201_sorted_results_adj_gin_804.pkl"]
kendall_tau = ensemble_pred_scores_and_calculate_kendall_tau(filenames)
print(f"SpearmanR correlation: {kendall_tau}")

