"""Bilinear model with l-bfgs solver"""

# Author: Vlad Niculae
# License: Simplified BSD

import numpy as np
from scipy import optimize
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot


def _bilinear_forward(U, V, X_left, X_right):
    return np.sum(safe_sparse_dot(X_left, U) * safe_sparse_dot(X_right, V),
                  axis=1)


def _bilinear_loss_grad(U, V, X_left, X_right, y, alpha):
    n_samples, n_features_left = X_left.shape
    n_components = V.shape[1]

    U = U.reshape((n_features_left, n_components))

    XlU = safe_sparse_dot(X_left, U)
    XrV = safe_sparse_dot(X_right, V)
    y_pred = np.sum(XlU * XrV, axis=1)

    # squared loss
    loss = np.mean((y_pred - y) ** 2)
    loss += alpha * ((U ** 2).sum() + (V ** 2).sum())
    loss *= 0.5

    # dloss_dy_hat
    grad_loss = (y_pred - y)[:, np.newaxis]
    grad_U = safe_sparse_dot(X_left.T, (grad_loss * XrV))
    grad_U /= n_samples
    grad_U += alpha * U ** 2

    return loss, grad_U.ravel()


def _bilinear_step(X_left, X_right, y, U, V, alpha=0.1, tol=1e-6, max_iter=1,
                   verbose=False):

    u_new, loss, info = optimize.fmin_l_bfgs_b(
        _bilinear_loss_grad,
        U.ravel(),
        fprime=None,
        args=(V, X_left, X_right, y, alpha),
        iprint=(verbose > 0) - 1,
        pgtol=tol,
        maxiter=max_iter)

    U_new = u_new.reshape(U.shape)

    v_new, loss, info = optimize.fmin_l_bfgs_b(
        _bilinear_loss_grad,
        V.ravel(),
        fprime=None,
        args=(U_new, X_right, X_left, y, alpha),
        iprint=(verbose > 0) - 1,
        pgtol=tol,
        maxiter=max_iter)

    V_new = v_new.reshape(V.shape)

    return loss, U_new, V_new


class BilinearRegressor(object):

    def __init__(self, alpha=1.0, n_components=10, random_state=0,
                 max_iter=5000, tol=1e-8, max_inner_iter=1, warm_start=False,
                 verbose=False):
        self.alpha = alpha
        self.n_components = n_components
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.max_inner_iter = max_inner_iter
        self.warm_start = warm_start
        self.verbose = verbose

    def fit(self, X_left, X_right, y):
        n_samples, n_features_left = X_left.shape
        n_samples, n_features_right = X_right.shape

        rng = check_random_state(self.random_state)

        if self.warm_start and hasattr(self, 'U_'):
            U = self.U_
        else:
            U = rng.randn(n_features_left, self.n_components)

        if self.warm_start and hasattr(self, 'V_'):
            V = self.V_
        else:
            V = rng.randn(n_features_right, self.n_components)

        old_loss = np.inf
        for it in range(1, self.max_iter + 1):
            loss, U, V = _bilinear_step(X_left, X_right, y, U, V, self.alpha,
                                        self.tol, self.max_inner_iter,
                                        self.verbose)

            if self.verbose > 1:
                print(" . iter {} loss {}".format(it, loss))

            if np.abs(old_loss - loss) < self.tol:
                if self.verbose:
                    print("Converged at iteration {}".format(it))
                break

            old_loss = loss

        self.U_ = U
        self.V_ = V

        return self

    def predict(self, X_left, X_right):
        return _bilinear_forward(self.U_, self.V_, X_left, X_right)


if __name__ == '__main__':
    rng = np.random.RandomState(42)
    X_left = rng.randn(100, 5)
    X_right = rng.randn(100, 5)

    true_U = rng.randn(5, 2)
    true_V = rng.randn(5, 2)

    y = _bilinear_forward(true_U, true_V, X_left, X_right)

    bl = BilinearRegressor(alpha=0.01, random_state=0, verbose=2)
    bl.fit(X_left, X_right, y)

    y_pred = bl.predict(X_left, X_right)
    print(np.mean((y_pred - y) ** 2))