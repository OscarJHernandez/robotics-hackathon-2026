"""Small statistics for comparing variants: Wilson interval, exact sign test, paired outcomes.

wilcoxon_p and spearman use scipy (imported lazily); the rest is plain Python."""

import math

GOAL = "goal"


def wilson(k, n, z=1.96):
    """Wilson score interval (low, high) for k successes in n trials. (0.0, 1.0) when n is 0."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def sign_test(wins, losses):
    """Two-sided exact binomial (p = 0.5) sign test on wins vs losses. Ties are not counted."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(a, b):
    """Compare two variants on the same episodes. a, b: {episode key: outcome}; only shared keys count.

    win = a reached the goal and b did not; loss = the reverse; tie = both or neither.
    """
    wins = losses = ties = 0
    for key in a.keys() & b.keys():
        a_goal, b_goal = a[key] == GOAL, b[key] == GOAL
        if a_goal and not b_goal:
            wins += 1
        elif b_goal and not a_goal:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties, "p": sign_test(wins, losses)}


def wilcoxon_p(differences):
    """Two-sided Wilcoxon signed-rank p-value for paired differences. Zero and NaN differences are dropped; 1.0
    when none are left (all-zero differences)."""
    diffs = [float(d) for d in differences if d == d and d != 0]
    if not diffs:
        return 1.0
    from scipy.stats import wilcoxon
    return float(wilcoxon(diffs).pvalue)


def spearman(a, b):
    """Spearman rank correlation of two equal-length sequences; None when undefined (fewer than 2 points or
    a constant input)."""
    a, b = list(a), list(b)
    if len(a) != len(b):
        raise ValueError("a and b differ in length")
    if len(a) < 2 or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)
