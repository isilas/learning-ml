# 01 | Classification from scratch

[Back to portfolio](../README.md) · [Implementation](fundamentals.py) · [Tests](../tests/test_algorithms.py)

## Question
How does a distance-based classifier make a prediction, and does a first-principles implementation agree with an established library?

## Approach
For a query point, calculate Euclidean distances to all training samples, sort, select the k closest samples and return their most frequent label. The original coursework implemented this procedure with NumPy and majority voting. Changing k from 3 to 5 changes the predicted class on its small example (1 to 0).

The new portfolio benchmark uses the scikit-learn bundled Wisconsin diagnostic breast cancer dataset (569 samples, 30 features), a stratified 75/25 split, k=5 and a scaler fitted on training data only. It compares predictions with scikit-learn's KNeighborsClassifier on the same split.

| Metric | New portfolio run |
|---|---:|
| Held-out samples | 143 |
| Accuracy | 97.90% |
| Balanced accuracy | 97.17% |
| Agreement with scikit-learn | 100% |

## Interpretation and limitations
Feature scale changes Euclidean distances. The parameter k trades local sensitivity against smoothing. A single split is not a robustness study, and this benchmark is a software verification exercise, not a clinical diagnostic system. The straightforward implementation scans and sorts training distances for each query; large datasets may benefit from indexing or approximate search.