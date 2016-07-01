import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
from lightning.impl.dataset_fast import get_dataset
from ..lbfgs import _bilinear_forward
from ..cd_fast import _cd_bilinear_epoch

from nose.tools import assert_almost_equal
from numpy.testing import assert_array_almost_equal


rng = np.random.RandomState(42)
X_left = rng.randn(20, 5)
X_right = rng.randn(20, 5)

true_U = rng.randn(5, 2)
true_V = rng.randn(5, 2)

W = np.dot(true_U, true_V.T)

# matrix inner product
y = np.diag(np.dot(np.dot(X_left, W), X_right.T))


def _bilinear_cd(U, V, X_left, X_right, y, alpha):
    n_samples, n_features_left = X_left.shape
    n_components = V.shape[1]

    XrV = safe_sparse_dot(X_right, V)

    viol = 0

    for j in range(n_features_left):
        for s in range(n_components):

            XlU = safe_sparse_dot(X_left, U)
            y_pred = np.sum(XlU * XrV, axis=1)

            # grad_loss = loss.dloss(y_pred, y)
            grad_loss = y_pred - y

            grad = np.dot(grad_loss * X_left[:, j], XrV[:, s])
            # grad /= n_samples
            grad += alpha * U[j, s]

            inv_step_size = np.dot(X_left[:, j] ** 2, XrV[:, s] ** 2)
            # inv_step_size /= np.sqrt(n_samples)
            inv_step_size += alpha

            update = grad / inv_step_size
            viol += np.abs(update)
            U[j, s] -= update

    XlU = safe_sparse_dot(X_left, U)
    y_pred = np.sum(XlU * XrV, axis=1)
    lv = 0.5 * np.sum((y_pred - y) ** 2)
    lv += 0.5 * alpha * (np.sum(U ** 2) + np.sum(V ** 2))

    return viol, lv


def test_epoch():
    U = rng.randn(*true_U.shape)
    U2 = U.copy()

    viol, lv = _bilinear_cd(U, true_V, X_left, X_right, y, 1.0)

    dataset = get_dataset(X_left, 'fortran')

    # precomputing for cython
    y_pred = _bilinear_forward(U2, true_V, X_left, X_right)
    XrV = safe_sparse_dot(X_right, true_V)
    VtGsq = safe_sparse_dot(XrV.T ** 2, X_left ** 2)
    v2 = _cd_bilinear_epoch(U2, dataset, XrV, y, y_pred, VtGsq, 1.0)

    assert_almost_equal(viol, v2)
    assert_array_almost_equal(U, U2)
