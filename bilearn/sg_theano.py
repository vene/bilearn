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

from lasagne.updates import adam  # could use sgd, adagrad, etc
from lasagne.objectives import binary_hinge_loss, binary_crossentropy


def safe_sparse_mul(X, Y):
    if hasattr(X, 'multiply'):
        return X.multiply(Y)
    else:
        return X * Y


class LowRankBilinear(object):

    def __init__(self, n_components=5, l2_reg=0.01, l2_reg_linear=0.01,
                 l2_reg_diag=0.01, max_iter=1000, random_state=0,
                 warm_start=False, fit_diag=True, fit_linear=True,
                 loss_func=None, update_rule=None, update_params=None):
        self.n_components = n_components
        self.l2_reg = l2_reg
        self.l2_reg_linear = l2_reg_linear
        self.l2_reg_diag = l2_reg_diag
        self.max_iter = max_iter
        self.random_state = random_state
        self.warm_start = warm_start
        self.fit_diag = fit_diag
        self.fit_linear = fit_linear
        self.loss_func = loss_func
        self.update_rule = update_rule
        self.update_params = update_params

    def fit(self, X_left, X_right, y):

        # TODO refactor as X and self.left_slice, self.right_slice
        # Somehow we have to make sure this works nicely with a FeatureUnion
        n_samples, n_features_left = X_left.shape
        n_samples_right, n_features_right = X_right.shape

        assert n_samples == n_samples_right

        fit_diag = self.fit_diag
        if fit_diag and n_features_left != n_features_right:
            warnings.warn("Cannot fit diagonal if spaces have diff. dim.")
            fit_diag = False

        rng = check_random_state(self.random_state)

        if not self.warm_start or (hasattr(self, 'U_') and self.U_ is None):
            U_init = rng.randn(n_features_left, self.n_components)
        else:
            U_init = self.U_

        if not self.warm_start or (hasattr(self, 'V_') and self.V_ is None):
            V_init = rng.randn(n_features_right, self.n_components)
        else:
            V_init = self.V_

        if self.update_rule is None:
            update_rule = adam
        else:
            update_rule = self.update_rule

        if self.update_params is None:
            update_params = {}
        else:
            update_params = self.update_params

        U = tn.shared(value=U_init, name='U')
        V = tn.shared(value=V_init, name='V')
        trainable_params = [U, V]

        if self.fit_linear:
            w_left = tn.shared(value=np.zeros(n_features_left), name='w_left')
            w_right = tn.shared(value=np.zeros(n_features_right),
                                name='w_right')
            trainable_params.extend([w_left, w_right])

        if fit_diag:
            diag = tn.shared(value=np.ones(n_features_left), name='diag')
            trainable_params.append(diag)

        if sp.issparse(X_left):
            X_left_tn = tsp.csr_matrix('X_left_tn')
            UX_left = tsp.dot(X_left_tn, U)
        else:
            X_left_tn = T.dmatrix('X_left_tn')
            UX_left = T.dot(X_left_tn, U)

        if sp.issparse(X_left):
            X_right_tn = tsp.csr_matrix('X_right_tn')
            UX_right = tsp.dot(X_right_tn, V)
        else:
            X_right_tn = T.dmatrix('X_right_tn')
            UX_right = T.dot(X_right_tn, V)

        y_tn = T.bvector('y')

        y_scores = T.batched_dot(UX_left, UX_right)

        if self.fit_linear:
            if sp.issparse(X_left_tn):
                wX_left = tsp.dot(X_left_tn, w_left)
            else:
                wX_left = T.dot(X_left_tn, w_left)

            if sp.issparse(X_right_tn):
                wX_right = tsp.dot(X_right_tn, w_right)
            else:
                wX_right = T.dot(X_right_tn, w_right)

            y_scores += wX_left + wX_right

        if fit_diag:
            if sp.issparse(X_left_tn) or sp.issparse(X_right_tn):
                XlXr = tsp.mul(X_left_tn, X_right_tn)
                y_scores += tsp.dot(XlXr, diag)
            else:
                XlXr = T.mul(X_left_tn, X_right_tn)
                y_scores += T.dot(XlXr, diag)

        # hinge loss
        loss = binary_hinge_loss(y_scores, y_tn).mean()

        # regularization
        loss += self.l2_reg * (T.sum(U ** 2) + T.sum(V ** 2))

        if self.fit_linear:
            loss += self.l2_reg_linear * (T.sum(w_left ** 2) +
                                          T.sum(w_right ** 2))

        if fit_diag:
            loss += self.l2_reg_diag * T.sum(diag ** 2)

        train_model = tn.function(inputs=[],
                                  outputs=loss,
                                  updates=update_rule(loss,
                                                      trainable_params,
                                                      **update_params),
                                  givens={X_left_tn: X_left,
                                          X_right_tn: X_right,
                                          y_tn: y})

        self.losses_ = []
        for _ in range(self.max_iter):
            self.losses_.append(train_model())

        self.U_ = U.eval()
        self.V_ = V.eval()

        if self.fit_linear:
            self.w_left_ = w_left.eval()
            self.w_right_ = w_right.eval()
        if fit_diag:
            self.diag_ = diag.eval()

    def decision_function(self, X_left, X_right):
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

    def predict(self, X_left, X_right):
        return self.decision_function(X_left, X_right) > 0


if __name__ == '__main__':
    rng = np.random.RandomState(42)
    X_left = rng.randn(100, 5)
    X_right = rng.randn(100, 5)

    true_U = rng.randn(5, 2)
    true_V = rng.randn(5, 2)
    true_diag = np.sign(rng.randn(5))

    y = np.sum(np.dot(X_left, true_U) * np.dot(X_right, true_V), axis=1)
    y += np.dot((X_left * X_right), true_diag)
    y = (y > 0).astype(np.int8)

    from time import time
    from itertools import product

    import matplotlib.pyplot as plt

    for fit_linear, fit_diag in product((False, True), (False, True)):
        print("fit_linear={}, fit_diag={}".format(fit_linear, fit_diag))
        lrbl = LowRankBilinear(n_components=5,
                               l2_reg=0.001,
                               l2_reg_linear=0.001,
                               l2_reg_diag=0.001,
                               fit_linear=fit_linear, fit_diag=fit_diag,
                               update_params={'learning_rate': 0.01},
                               # max_iter=10000,
                               random_state=0)
        t0 = time()
        # lrbl.fit(sp.csr_matrix(X_left), sp.csr_matrix(X_right), y)
        lrbl.fit(X_left, X_right, y)
        t0 = time() - t0
        y_pred_train = lrbl.predict(X_left, X_right)

        X_left_val = rng.randn(100, 5)
        X_right_val = rng.randn(100, 5)
        y_val = np.sum(np.dot(X_left_val, true_U) *
                       np.dot(X_right_val, true_V), axis=1)

        y_val += np.dot((X_left_val * X_right_val), true_diag)

        y_val = y_val > 0
        y_pred = lrbl.predict(X_left_val, X_right_val)

        plt.semilogy(lrbl.losses_,label="fit_linear={}, fit_diag={}".format(
           fit_linear, fit_diag))

        print("\t{:.2f}s".format(t0))
        print("\tTrain accuracy: {:.2f}".format(np.mean(y_pred_train == y)))
        print("\tTest accuracy: {:.2f}".format(np.mean(y_pred == y_val)))

    plt.legend()
plt.show()