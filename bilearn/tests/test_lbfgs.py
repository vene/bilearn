import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy import optimize
from scipy import sparse as sp

from bilearn.lbfgs import BilinearRegressor, _bilinear_forward
from bilearn.lbfgs import _bilinear_loss_grad

rng = np.random.RandomState(42)
X_left = rng.randn(20, 5)
X_right = rng.randn(20, 5)

true_U = rng.randn(5, 2)
true_V = rng.randn(5, 2)

W = np.dot(true_U, true_V.T)

# matrix inner product
y = np.diag(np.dot(np.dot(X_left, W), X_right.T))


def test_forward():
    """Test that predictions are computed correctly"""
    y_fwd = _bilinear_forward(true_U, true_V, X_left, X_right)
    assert_array_almost_equal(y_fwd, y)


def test_learn():
    bl = BilinearRegressor(alpha=0).fit(X_left, X_right, y)
    y_pred = bl.predict(X_left, X_right)
    assert_array_almost_equal(y_pred, y, decimal=2)


def test_logistic_loss_and_grad():
    X_left_sp = X_left.copy()
    X_left_sp[X_left_sp < .1] = 0
    X_left_sp = sp.csr_matrix(X_left_sp)

    X_right_sp = X_right.copy()
    X_right_sp[X_right_sp < .1] = 0
    X_right_sp = sp.csr_matrix(X_right_sp)

    U = np.zeros_like(true_U)
    V = np.zeros_like(true_V)

    for (Xl, Xr) in ((X_left, X_right), (X_left_sp, X_right_sp)):
        _, grad = _bilinear_loss_grad(U, V, Xl, Xr, y, alpha=1)
        approx_grad = optimize.approx_fprime(
            U.ravel(), lambda u: _bilinear_loss_grad(u, V, Xl, Xr, y,
                                             alpha=1)[0], 1e-3
        )
        assert_array_almost_equal(grad, approx_grad, decimal=2)
