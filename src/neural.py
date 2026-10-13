"""Neural network: a small ensemble of multi-layer perceptrons (feed-forward networks trained by back-propagation).

Each member is a 2-hidden-layer network (32 -> 16 ReLU units, L2 penalty, Adam, early stopping on a held-out 10 %
of the training fold). Members differ only in their random initialisation / validation split; averaging their
probabilities reduces variance, which matters for a small tabular data set with a rare positive class.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier


class MLPEnsemble(ClassifierMixin, BaseEstimator):
    def __init__(self, n_members=5, hidden=(32, 16), alpha=0.1, max_iter=200, random_state=0):
        self.n_members = n_members; self.hidden = hidden; self.alpha = alpha
        self.max_iter = max_iter; self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = [MLPClassifier(self.hidden, alpha=self.alpha, early_stopping=True, validation_fraction=0.1,
                                      n_iter_no_change=10, max_iter=self.max_iter, random_state=self.random_state + i).fit(X, y)
                        for i in range(self.n_members)]
        for m in self.models_:                      # the RNG is only needed while training; dropping it keeps the saved
            m.__dict__.pop("_random_state", None)   # model loadable across numpy versions (no MT19937 pickle error)
        return self

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.models_], axis=0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def architecture(self, n_inputs):
        """Layer sizes input -> hidden -> output, for display."""
        return [n_inputs, *self.hidden, 1]
