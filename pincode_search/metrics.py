import numpy as np


def recall_at_k(true_scores, got_scores, k=10, eps=1e-5):
    """Tie-aware recall: a result counts if it scores at least as high as the k-th true neighbour."""
    if len(true_scores) == 0:
        return 1.0
    wanted = min(k, len(true_scores))
    cutoff = true_scores[wanted - 1] - eps
    hits = int(np.sum(np.asarray(got_scores[:k]) >= cutoff))
    return min(hits, wanted) / wanted


def percentile_ms(seconds, q):
    return float(np.percentile(np.asarray(seconds) * 1000, q))
