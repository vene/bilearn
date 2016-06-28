import numpy as np
from numpy.testing import assert_array_almost_equal

from bilearn.sg_theano import BilinearRegressorSG

rng = np.random.RandomState(42)
X_left = rng.randn(20, 5)
X_right = rng.randn(20, 5)

true_U = rng.randn(5, 2)
true_V = rng.randn(5, 2)

W = np.dot(true_U, true_V.T)

# matrix inner product
y = np.diag(np.dot(np.dot(X_left, W), X_right.T))


def test_learn():
    bl = BilinearRegressorSG(alpha=0, max_iter=10000).fit(X_left, X_right, y)
    y_pred = bl.predict(X_left, X_right)
    assert_array_almost_equal(y_pred, y, decimal=3)