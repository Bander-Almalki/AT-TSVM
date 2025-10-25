import os
import numpy as np
import pandas
import pickle
import math

def calculate_top_l_precisions(pred_scores, true_labels):
    """
    Calculate Top-L, L/2, L/5, and L/10 precision for a contact map.

    Args:
        pred_scores (np.ndarray): 2D array of predicted contact probabilities (shape L x L).
        true_labels (np.ndarray): 2D array of ground truth binary labels (same shape).

    Returns:
        dict: Precision scores at Top-L, Top-L/2, Top-L/5, Top-L/10.
    """
    L = int(math.sqrt(pred_scores.shape[0]*2))
    assert pred_scores.shape == true_labels.shape

    # Collect upper triangle indices (i < j)
    #indices = pred_scores[:,[0,1]]
    scores = pred_scores
    labels =true_labels

    # Sort predictions by score descending
    sorted_indices = np.argsort(scores)[::-1]
    labels_sorted = labels[sorted_indices]

    def precision_at_k(k):
        if k == 0:
            return 0.0
        top_k = labels_sorted[:k]
        return np.sum(top_k) / k

    return {
        "Top-L": precision_at_k(L),
        "Top-L/2": precision_at_k(L // 2),
        "Top-L/5": precision_at_k(L // 5),
        "Top-L/10": precision_at_k(L // 10)
    }

