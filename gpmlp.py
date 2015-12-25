import math
import numpy as np
from  sklearn import datasets
from sklearn import preprocessing
from sknn.mlp import Regressor, Layer
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import Lasso, Ridge, LassoCV

import pickle
from pylab import *
import synthetic_data

from mlp import MLP
from rd_gp_mlp import rd_mlp

def metrics(pred, y, name):
    if name.lower() == 'mse':
        return mse(pred, y)
    elif name.lower() == 'auc':
        def auc(pred, y):

            n = 0
            correct = 0

            for i in range(len(y)):
               for j in range(i+1, len(y)):
                    if (pred[i] - pred[j]) * (y[i] - y[j]) > 0:
                        correct += 1

                    n += 1

            return correct * 1.0 / n

        return auc(pred, y)

def activation(x, name):

    if name.lower() == 'relu':
        relu_x = x.copy()
        relu_x[x <= 0] = 0
        return relu_x
    elif name.lower() == "sigmoid":
        return 1 / (1 + np.exp(-x))

class gp_mlp:

    def __init__(self, hidden_len=3, activation_func="RELU", reg_v=0, reg_w=0, momentum=0.5, init_param=1, norm_style=None, metrics_func = "MSE"):

        self.hidden_len = hidden_len
        self.activation_func = activation_func

        self.hidden_unit = np.zeros(hidden_len)
        self.init_param = init_param

        self.active_d = []
        self.inactive_d = []

        self.norm_style = norm_style
        self.activation_func = activation_func
        self.metrics_func = metrics_func

        self.reg_v = reg_v
        self.reg_w = reg_w
        self.momentum=momentum

    def gp_fit(self, X, y, gp_lambda=0.1, max_iter=50, lr=0.01, debug=False, tst_X=None, tst_y=None, lasso_model=None):

        assert X.ndim  == 2
        assert X.shape[0] == len(y)

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        tst_X = np.hstack([np.ones((tst_X.shape[0], 1)), tst_X])

        self.N = X.shape[0]
        self.input_dim = X.shape[1]
        self.gp_lambda = gp_lambda

        self.init_lr = lr
        self.lr = self.init_lr


        self.active_d = [0]
        self.inactive_d = range(1, self.input_dim)

        self._init_gp_param()

        mses = []
        tst_mses = []
        is_add = True

        #Training by iteratively
        while(is_add):

            #Train MLP using current active _d
            it_mses = []
            old_v = self.V.copy()
            old_w = self.W.copy()

            #save the gradient norm of each iteration
            grad_w_0 = []

            #while True:
            for tttttt in range(50):
                sgd_shuffle = np.arange(X.shape[0])
                np.random.shuffle(sgd_shuffle)
                batch_size = 1 #X.shape[0]

                for sgd_i in range(int(math.ceil(X.shape[0] / batch_size))):
                    grad_w = self.gp_update(X[(sgd_i * batch_size):((sgd_i+1) * batch_size), :],y[(sgd_i * batch_size):((sgd_i+1) * batch_size)])

                #Save the gradient norm change
                # try:
                #     grad_w_0.append(np.linalg.norm(grad_w,axis=1)[0])
                # except:
                #     grad_w_0.append(np.linalg.norm(grad_w,axis=1))t
                it_mses.append(metrics(self.gp_predict(X), y, self.metrics_func))
                if len(it_mses) > 5:
                    if ((it_mses[-5] - it_mses[-1]) < 0):
                        break
                if len(it_mses) > 30:
                    if (((it_mses[-20] - it_mses[-1]) / it_mses[-20]) < 0.00001):
                         break

            #Show the gradient norm
            # figure()
            # plot(grad_w_0, "*-", linewidth=3)
            # show()

            if True:
                figure()
                title("Current Active Features:%d" % (len(self.active_d)))
                plot(it_mses, "*-", linewidth=3)
                savefig("fig/%d.jpg" % len(self.active_d))

            mses.append(metrics(self.gp_predict(X), y, self.metrics_func))
            tst_mses.append(metrics(self.gp_predict(tst_X), tst_y, self.metrics_func))

            #########################################
            #Test using concat features.
            # new_tr_x = np.hstack([X, self.apply(X)])
            # new_tst_x = np.hstack([tst_X, self.apply(tst_X)])
            # concat_lr = Lasso(alpha=0.25, fit_intercept=True, normalize=False)
            # concat_lr.fit(new_tr_x, y)
            # tst_mses.append(metrics(tst_y, concat_lr.predict(new_tst_x), self.metrics_func))
            #tst_mses.append(metrics(tst_y, lasso_model.predict(tst_X[:, 1::]) + self.gp_predict(tst_X), self.metrics_func))

            #Test Using MLP
            nn = Regressor(
                    layers = [
                        Layer("Rectifier", units=100),
                        Layer("Sigmoid", units=50),
                        Layer("Linear")
                    ],
                    learning_rate=0.00001,
                    n_iter=50,
                    regularize="L2",
                    weight_decay=0.005
                    )
            nn.fit(X[:, self.active_d], y)#-ls.predict(tr_X))
            tr_pred = nn.predict(X[:, self.active_d]).flatten() #+ ls.predict(tr_X).flatten()
            tst_pred = nn.predict(tst_X[:, self.active_d]).flatten() #+ ls.predict(tst_X).flatten()

            print "SKNN Tr MSE:%f, Tst MSE:%f" % (metrics(tr_pred, y, self.metrics_func), metrics(tst_pred, tst_y, self.metrics_func))
            #print "Total Selected:%d, Hidden Selected:%d, Sparsity Tr:%f, Sparsity Tst:%f" % (np.sum(concat_lr.coef_ != 0), np.sum(concat_lr.coef_[X.shape[1]::] != 0), np.mean(self.apply(X) == 0), np.mean(self.apply(tst_X) == 0))
            #print "Iter:%d, Tr MSE:%f, Tst MSE:%f, V[0]:%f, SP RELU:%f" % (len(self.active_d), mses[-1], mse(tst_y, self.gp_predict(tst_X)), self.V[0], np.mean(self.hidden_unit == 0))

            # if len(mses) > 2:
            #     #if mses[-1] > mses[-2] or it_mses[-1] > it_mses[0]:
            #     if mses[-1] > mses[-2]:
            #         #Do not Incorporate the features
            #         self.active_d.pop(-1)
            #         self.W = self.W[0:-1, :]
            #         self.vW = self.vW[0:-1, :]
            #         tst_mses.pop(-1)
            #         mses.pop(-1)
            #         print "Error Add Dim"
            #     else:
            #         if True:#len(self.active_d) % 5 == 1:
            #             figure()
            #             title("Current Active Features:%d" % (len(self.active_d)))
            #             plot(it_mses, "*-", linewidth=3)
            #             #show()
            #             savefig("fig/%d.jpg" % len(self.active_d))
            #             print "Iter:%d, Tr MSE:%f, Tst MSE:%f, SP RELU:%f" % (len(self.active_d), mses[-1], tst_mses[-1], np.mean(self.hidden_unit == 0))
            # else:
            #     print "Iter:%d, Tr MSE:%f, Tst MSE:%f, SP RELU:%f" % (len(self.active_d), mses[-1], tst_mses[-1], np.mean(self.hidden_unit == 0))
            print "Iter:%d, Tr MSE:%f, Tst MSE:%f, SP RELU:%f" % (len(self.active_d), mses[-1], tst_mses[-1], np.mean(self.hidden_unit == 0))

            #Add new features
            is_add = self.gp_add_feature(X,y)

    def _init_gp_param(self):

        if self.init_param.lower() == "optimal":
            self.W = np.random.uniform(low=-4*math.sqrt(6.0/(2 * self.hidden_len)), high=4*math.sqrt(6.0/(2 * self.hidden_len)), size=(len(self.active_d), self.hidden_len))
            self.V = np.random.uniform(low=-4*math.sqrt(6.0/(self.hidden_len)), high=4*math.sqrt(6.0/(self.hidden_len)), size=(self.hidden_len+1))
            self.lW = np.zeros(self.input_dim)
        else:
            self.W = np.random.normal(loc=0, scale=self.init_param, size=(len(self.active_d), self.hidden_len)) # Active Input -> Hidden
            self.V = np.random.normal(loc=0, scale=self.init_param, size=(self.hidden_len+1)) #Hidden to Output
            self.lW = np.zeros(self.input_dim)

        # self.W = np.zeros((len(self.active_d), self.hidden_len))
        # self.V = np.zeros(self.hidden_len+1)

        #Record gradient updates considering momentum
        self.vW = np.zeros(self.W.shape)
        self.vV = np.zeros(self.V.shape)

    def gp_update(self, X, y):

    	if len(self.active_d) == 0:
    	    return

        pred = self.gp_predict(X)
        err = (y - pred)
        active_X = X[:, self.active_d]

        #Gradient of v
        grad_v = - np.dot(err, np.hstack([np.ones((self.hidden_unit.shape[0], 1)), self.hidden_unit])) / self.N

        #Gradient of W

        #v_tile = np.tile(self.V, (self.N, 1)) #N * k
        v_tile = np.tile(self.V[1::], (X.shape[0], 1))
        err_tile = np.tile(err.reshape(len(err), 1), (1, self.hidden_len)) #N * k

        if self.activation_func.lower()=="relu":
            I_kl  = self.hidden_unit.copy() #N * k
            I_kl[self.hidden_unit == 0] = 0.1
            I_kl[self.hidden_unit != 0] = 1
            err_I_vj = err_tile * (I_kl * v_tile)
        elif self.activation_func.lower()=="sigmoid":
            err_I_vj = err_tile * (self.hidden_unit * (1 - self.hidden_unit)) * v_tile

        grad_w = - np.dot(active_X.T, err_I_vj) / self.N

        #Regularization
        grad_w = grad_w + self.reg_w * self.W
        grad_v = grad_v + self.reg_v * self.V

        #Momentum
        self.vW = self.momentum * self.vW + self.lr * grad_w
        self.vV = self.momentum * self.vV + self.lr * grad_v

        #Gradient Update
        self.W = self.W - self.vW
        self.V = self.V - self.vV

        self.lW = self.lW - self.lr * (-np.dot(err, X).flatten() + self.reg_v * self.lW)

        #Sparse(Lasso) Regularization
        # threshold_V = (np.abs(self.V) - self.lr * 1)
        # threshold_V[threshold_V < 0] = 0
        # self.V = np.sign(self.V) * threshold_V

        # threshold_W = (np.abs(self.W) - self.lr * 1)
        # threshold_W[threshold_W < 0] = 0
        # self.W = np.sign(self.W) * threshold_W

        return grad_w

    def gp_add_feature(self, X, y):

        if len(self.inactive_d) == 0:
            return False

        pred = self.gp_predict(X)
        err = (y - pred)

        #Gradient of W
        v_tile = np.tile(self.V[1::], (self.N, 1)) #N * k
        err_tile = np.tile(err.reshape(len(err), 1), (1, self.hidden_len)) #N * k

        if self.activation_func.lower() == "relu":
            I_kl  = self.hidden_unit.copy() #N * k
            I_kl[self.hidden_unit == 0] = 0.1
            I_kl[self.hidden_unit != 0] = 1
            err_I_vj = err_tile * (I_kl * v_tile)
        elif self.activation_func.lower() == "sigmoid":
            err_I_vj = err_tile * (self.hidden_unit * (1 - self.hidden_unit)) * v_tile

        grad_w = - np.dot(X.T, err_I_vj) / self.N

        inactive_norm_grad = np.linalg.norm(grad_w, axis=1, ord=self.norm_style)[self.inactive_d]
        inactive_id = np.argmax(inactive_norm_grad)

        #Show Gradient Distribution
        # plot(inactive_norm_grad * self.lr, "*-", linewidth=3)
        # title("Distributed of Norm of Gradient")
        # show()
        # print "Largest Gradient:%f" % (inactive_norm_grad[inactive_id] * self.lr)

        if inactive_norm_grad[inactive_id] * self.init_lr <= self.gp_lambda:
            #End Training
            return False

        #Add Feature
        self.active_d.append(self.inactive_d[inactive_id])
        self.inactive_d.pop(inactive_id)

        if self.init_param.lower() == 'optimal':
            self.W = np.vstack([self.W, np.random.uniform(low=-4*math.sqrt(6.0/(2 * self.hidden_len)), high=4*math.sqrt(6.0/(2 * self.hidden_len)), size=(1, self.hidden_len))])
        else:
            self.W = np.vstack([self.W, np.random.normal(loc=0, scale=self.init_param, size=(1, self.hidden_len))])
        self.vW = np.vstack([self.vW, np.zeros((1, self.hidden_len))])

        return True


    def gp_predict(self, X):

        "Forward Neural Network that predict the label"

    	if len(self.active_d) == 0:
    	    return np.zeros(X.shape[0])

        self.hidden_unit = activation(np.dot(X[:, self.active_d], self.W), self.activation_func) #N * k
        y_pred = np.dot(np.hstack([np.ones((self.hidden_unit.shape[0], 1)), self.hidden_unit]), self.V) #N

        #Cross Level Connection from hidden to Output
        y_pred += np.dot(X, self.lW)

        return y_pred

    def apply(self, X):
        act = activation(np.dot(X[:, self.active_d], self.W), self.activation_func)

        return act - np.tile(np.mean(act, 0), (act.shape[0], 1))



