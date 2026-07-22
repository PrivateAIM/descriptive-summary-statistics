# Distributed Variance Aggregation Fix

## Problem

The original aggregator computed the pooled variance across nodes using the
raw-moment identity:

```
Var(X) = E[X²] − (E[X])²
```

This identity is exact only for the **population** variance (ddof=0). When
each node reports a **sample** variance (ddof=1, divided by n−1), and the
aggregator reconstructs E[X²] from those local statistics, the result is
biased: it effectively applies an N−K denominator (where K is the number of
nodes) rather than the correct N−1.

The bug was most visible when node means differ significantly but within-node
spread is small. For example, two nodes with values {0, 0} and {10, 10}:

- True pooled sample variance: **100/3 ≈ 33.33**
- Old formula result: **25.0** (wrong — off by 25%)

## Fix

Replaced the identity with the exact **Chan–Golub–LeVeque parallel
decomposition** (see citation below).

### Formula

Given K nodes, each reporting its local count nᵢ, local mean x̄ᵢ, and local
sample variance sᵢ² (ddof=1):

1. Compute the global weighted mean:

```
x̄  =  (Σ nᵢ · x̄ᵢ) / N        where N = Σ nᵢ
```

2. Reconstruct the total sum of squared deviations from x̄:

```
SS_total  =  Σ [ (nᵢ − 1)·sᵢ²  +  nᵢ·(x̄ᵢ − x̄)² ]
```

The first term `(nᵢ − 1)·sᵢ²` is each node's own within-node sum of squares
(undoing the ddof=1 division). The second term `nᵢ·(x̄ᵢ − x̄)²` is the
between-node contribution — how far that node's mean sits from the global
mean, weighted by its size.

3. Divide by N−1 to get the pooled sample variance:

```
s²  =  SS_total / (N − 1)
```

This is **not** the same as the ANOVA pooled-within-group variance, which
divides by N−K. That formula measures average within-group spread and discards
between-group variation. Here we want the sample variance of the full
concatenated dataset, so the denominator is always N−1.

### Implementation

```python
# data_report/analyze.py

def combine_node_variances(node_stats, global_mean: float) -> float:
    node_stats = list(node_stats)
    total_n = sum(n for n, _, _ in node_stats)
    if total_n <= 1:
        return 0.0

    sum_squares = sum(
        (n - 1) * var + n * (mean - global_mean) ** 2
        for n, mean, var in node_stats
    )
    return sum_squares / (total_n - 1)
```

`node_stats` is an iterable of `(n, mean, var)` tuples — one per node, where
`var` is the node's local sample variance (ddof=1).

The aggregator loop (`analyze.py`, numeric-statistics section) collects these
tuples per column and calls `combine_node_variances` after computing the
global mean.

## Tests

Five tests were added in `tests/test_new_modules.py` under
`TestCombineNodeVariances`:

| Test | What it checks |
|------|----------------|
| `test_matches_pooled_ground_truth_when_node_means_differ` | The {0,0}/{10,10} counterexample — all variance comes from between-node term; expected value 100/3 is derived from raw data independently |
| `test_matches_numpy_ground_truth_on_random_data` | 300 random samples split into three uneven chunks; result must match `numpy.var(..., ddof=1)` on the full array — the strongest end-to-end check |
| `test_single_node_reduces_to_its_own_variance` | With one node, the function must return that node's own variance unchanged |
| `test_single_total_sample_returns_zero` | N=1 means no degrees of freedom; result must be 0 |
| `test_no_samples_returns_zero` | Empty input must not raise and must return 0 |

All 69 tests in the suite pass.

## Citation

> Chan, T. F., Golub, G. H., & LeVeque, R. J. (1979).
> *Updating Formulae and a Pairwise Algorithm for Computing Sample Variances.*
> Technical Report STAN-CS-79-773, Stanford University, Department of Computer Science.

A concise description of the parallel algorithm (the form used here) is also
available on Wikipedia:
https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
