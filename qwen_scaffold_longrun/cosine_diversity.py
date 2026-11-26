"""Within-group cosine diagnostics; reward direction/weight are not decided here.

Pass embeddings of final answers for ONE task/scaffold group. No diagonal
comparisons, task mixing, or embedding of tool logs/reasoning is performed.
"""
import math


def cosine_summary(embeddings, threshold=0.5):
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError('Cosine threshold must be finite and in [-1, 1]')
    if len(embeddings) < 2:
        raise ValueError('At least two answers are required')
    dimension = len(embeddings[0])
    if dimension == 0 or any(len(v) != dimension for v in embeddings):
        raise ValueError('Embeddings must have a common nonzero dimension')
    normalized = []
    for vector in embeddings:
        if any(not math.isfinite(x) for x in vector):
            raise ValueError('Non-finite embedding')
        norm = math.hypot(*vector)
        if not math.isfinite(norm) or norm == 0:
            raise ValueError('Invalid embedding norm')
        normalized.append([x/norm for x in vector])
    pairs = [[] for _ in embeddings]
    for i in range(len(normalized)):
        for j in range(i):
            sim = max(-1., min(1., math.fsum(x*y for x,y in zip(normalized[i], normalized[j]))))
            pairs[i].append(sim)
            pairs[j].append(sim)
    return [
        {'mean_cosine_similarity':math.fsum(sims)/len(sims),
         'nearest_cosine_similarity':max(sims),
         'fraction_similarity_below_threshold':sum(s < threshold for s in sims)/len(sims),
         'fraction_similarity_above_threshold':sum(s > threshold for s in sims)/len(sims),
         'peer_count':len(sims)}
        for sims in pairs
    ]
