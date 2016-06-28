# low-rank bilinear regression using theano (supports sparse inputs)

# predicts f(x_left, x_right) = x_left' UV' x_right
# Reference:
#  Generalised Bilinear Regression
#  K. Ruben Gabriel
#  Source: Biometrika, Vol. 85, No. 3 (Sep., 1998), pp. 689-700
#  Stable URL: http://www.jstor.org/stable/2337396

# Author: Vlad Niculae <vlad@vene.ro>
# License: Simplified BSD

import warnings

import numpy as np
from scipy import sparse as sp

from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot

import theano as tn
import theano.tensor as T
from theano import sparse as tsp
from theano.sparse.basic import _is_sparse_variable as _tn_is_sparse

from lasagne.updates import adam  # could use sgd, adagrad, etc
from lasagne.objectives import binary_hinge_loss, binary_crossentropy


def safe_sparse_mul(X, Y):
    if hasattr(X, 'multiply'):
        return X.multiply(Y)
    else:
        return X * Y


def theano_safe_sparse_dot(X, Y):
    if _tn_is_sparse(X) or _tn_is_sparse(Y):
        return tsp.dot(X, Y)
    else:
        return T.dot(X, Y)


class BilinearSG(object):

    def _get_low_rank_term(self, X_left, X_right, U_init, V_init):
        U = tn.shared(value=U_init, name='U')
        V = tn.shared(value=V_init, name='V')

        UX_left = theano_safe_sparse_dot(X_left, U)
        VX_right = theano_safe_sparse_dot(X_right, V)

        y_pred = T.batched_dot(UX_left, VX_right)

        return y_pred, [U, V]

    def _get_linear_term(self, X_left, X_right, w_left_init, w_right_init):
        n_features_left = X_left.shape[1]
        n_features_right = X_right.shape[1]
        w_left = tn.shared(value=w_left_init, name='w_left')
        w_right = tn.shared(value=w_right_init, name='w_right')

        wX_left = theano_safe_sparse_dot(X_left, w_left)
        wX_right = theano_safe_sparse_dot(X_right, w_right)
        y_pred = wX_left + wX_right

        return y_pred, [w_left, w_right]

    def _get_diagonal_term(self, X_left, X_right, diag_init):
        diag = tn.shared(value=diag_init, name='diag')

        if _tn_is_sparse(X_left) or _tn_is_sparse(X_right):
            XlXr = tsp.mul(X_left, X_right)
            y_pred = tsp.dot(XlXr, diag)
        else:
            XlXr = T.mul(X_left, X_right)
            y_pred = T.dot(XlXr, diag)

        return y_pred, [diag]


class BilinearRegressorSG(BilinearSG):

    def __init__(self, n_components=5, l2_reg=0.01, max_iter=1000,
                 random_state=0, warm_start=False, fit_diag=True,
                 fit_linear=True, update_rule=None, update_params=None):
        self.n_components = n_components
        self.l2_reg = l2_reg
        self.max_iter = max_iter
        self.random_state = random_state
        self.warm_start = warm_start
        self.fit_diag = fit_diag
        self.fit_linear = fit_linear
        self.update_rule = update_rule
        self.update_params = update_params

    def fit(self, X_left, X_right, y):

        # TODO refactor as X and self.left_slice, self.right_slice
        # Somehow we have to make sure this works nicely with a FeatureUnion
        n_samples, n_features_left = X_left.shape
        n_samples_right, n_features_right = X_right.shape

        assert n_samples == n_samples_right
        if self.fit_diag and n_features_left != n_features_right:
            raise ValueError("Cannot fit diagonal term if spaces have "
                             "different number of features.")

        rng = check_random_state(self.random_state)

        # initialize all params if warm start is on
        if self.warm_start and hasattr(self, 'U_'):
            U_init = self.U_
        else:
            U_init = rng.randn(n_features_left, self.n_components)

        if self.warm_start and hasattr(self, 'V_'):
            V_init = self.V_
        else:
            V_init = rng.randn(n_features_right, self.n_components)

        if self.warm_start and hasattr(self, 'w_left_'):
            w_left_init = self.w_left_
        else:
            w_left_init = np.zeros(n_features_left)

        if self.warm_start and hasattr(self, 'w_right_'):
            w_right_init = self.w_right_
        else:
            w_right_init = np.zeros(n_features_right)

        if self.warm_start and hasattr(self, 'diag_'):
            diag_init = self.diag_
        else:
            diag_init = np.ones(n_features_left)

        if self.update_rule is None:
            update_rule = adam
        else:
            update_rule = self.update_rule

        if self.update_params is None:
            update_params = {}
        else:
            update_params = self.update_params

        if sp.issparse(X_left):
            X_left_tn = tsp.csr_matrix('X_left_tn')
        else:
            X_left_tn = T.dmatrix('X_left_tn')

        if sp.issparse(X_right):
            X_right_tn = tsp.csr_matrix('X_right_tn')
        else:
            X_right_tn = T.dmatrix('X_right_tn')

        y_tn = T.dvector('y')

        y_pred, vars = self._get_low_rank_term(X_left_tn, X_right_tn,
                                               U_init, V_init)
        U, V = vars

        if self.fit_linear:
            y_linear, vars_linear = self._get_linear_term(
                X_left_tn, X_right_tn, w_left_init, w_right_init)

            y_pred += y_linear
            vars += vars_linear
            w_left, w_right = vars_linear

        if self.fit_diag:
            y_diag, vars_diag = self._get_diagonal_term(
                X_left_tn, X_right_tn, diag_init)

            y_pred += y_diag
            vars += vars_diag
            diag, = vars_diag

        # squared loss
        loss = T.mean((y_pred - y_tn) ** 2)

        # hinge loss
        # loss = binary_hinge_loss(y_scores, y_tn).mean()

        # regularization
        for var in vars:
            loss += self.l2_reg * T.sum(var ** 2)

        train_model = tn.function(inputs=[X_left_tn, X_right_tn, y_tn],
                                  outputs=loss,
                                  updates=update_rule(loss, vars,
                                                      **update_params))

        self.losses_ = []

        for _ in range(self.max_iter):
            self.losses_.append(train_model(X_left, X_right, y))

        self.U_ = U.eval()
        self.V_ = V.eval()

        if self.fit_linear:
            self.w_left_ = w_left.eval()
            self.w_right_ = w_right.eval()

        if fit_diag:
            self.diag_ = diag.eval()

        return self

    def predict(self, X_left, X_right):
        y_pred = np.sum(safe_sparse_dot(X_left, self.U_) *
                        safe_sparse_dot(X_right, self.V_),
                        axis=1)

        if self.fit_linear:
            y_pred += safe_sparse_dot(X_left, self.w_left_)
            y_pred += safe_sparse_dot(X_right, self.w_right_)

        if self.fit_diag and hasattr(self, 'diag_'):
            y_pred += safe_sparse_dot(safe_sparse_mul(X_left, X_right),
                                      self.diag_)
        return y_pred


if __name__ == '__main__':
    rng = np.random.RandomState(42)
    X_left = rng.randn(100, 5)
    X_right = rng.randn(100, 5)

    true_U = rng.randn(5, 2)
    true_V = rng.randn(5, 2)
    true_diag = np.sign(rng.randn(5))

    y = np.sum(np.dot(X_left, true_U) * np.dot(X_right, true_V), axis=1)
    y += np.dot((X_left * X_right), true_diag)
    # y += 0.01 * rng.randn(100)

    from time import time
    from itertools import product

    import matplotlib.pyplot as plt

    for fit_linear, fit_diag in product((False, True), (False, True)):

        print("fit_linear={}, fit_diag={}".format(fit_linear, fit_diag))

        lrbl = BilinearRegressorSG(n_components=10,
                                   l2_reg=0.01,
                                   fit_linear=fit_linear,
                                   fit_diag=fit_diag,
                                   max_iter=20000,
                                   random_state=0)
        t0 = time()

        lrbl.fit(X_left, X_right, y)
        t0 = time() - t0
        y_pred_train = lrbl.predict(X_left, X_right)

        X_left_val = rng.randn(100, 5)
        X_right_val = rng.randn(100, 5)
        y_val = np.sum(np.dot(X_left_val, true_U) *
                       np.dot(X_right_val, true_V), axis=1)

        y_val += np.dot((X_left_val * X_right_val), true_diag)

        y_pred = lrbl.predict(X_left_val, X_right_val)

        plt.semilogy(lrbl.losses_, label="fit_linear={}, fit_diag={}".format(
           fit_linear, fit_diag))

        print("\t{:.2f}s".format(t0))
        print("\tTrain MSE: {:.5f}".format(np.mean((y_pred_train - y) ** 2)))
        print("\tTest MSE: {:.5f}".format(np.mean((y_pred - y_val) ** 2)))

    plt.legend()
    plt.show()