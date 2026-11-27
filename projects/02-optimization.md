# 02 | Optimization and decision-tree foundations

[Back to portfolio](../README.md) · [Implementation](fundamentals.py) · [Tests](../tests/test_algorithms.py)

## Question
Can an objective be differentiated and checked independently, and how does a tree decide where to split?

## Logistic regression with regularization
For logits z = Xβ and probabilities p = sigmoid(z), the objective maximizes average Bernoulli log-likelihood minus L1 and L2 penalties on the non-intercept coefficients. The analytical gradient is Xᵀ(y − p)/n with the penalty derivatives subtracted from the non-intercept entries. L1 encourages sparsity; L2 discourages large weights. At zero, L1 uses a subgradient rather than an ordinary derivative.

Original coursework saved the following numerical values; the new reproduction obtains the same values and checks them using central finite differences away from the L1 kink:

| Quantity | Value |
|---|---:|
| Penalized log-likelihood | −0.513996 |
| Intercept gradient | −0.057973 |
| Slope gradient | 0.015985 |

## Decision-tree splits
Gini impurity is 1 minus the sum of squared class probabilities. The gain from a split is parent impurity minus the sample-weighted child impurities. The original notebook compared several candidate thresholds. The new implementation verifies that a perfect split of four equally divided observations has gain 0.5.

## Interpretation and limitations
These examples verify the mechanics of gradient and split calculations, not an end-to-end optimized logistic regression model or a trained decision tree. Finite differences test local derivatives; they do not prove global convergence.