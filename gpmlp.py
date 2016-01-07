import math
import numpy as np
from  sklearn import datasets
from sklearn import preprocessing
from sknn.mlp import Regressor, Layer, Classifier
from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import Lasso, Ridge, LassoCV, LogisticRegressionCV as LogRCV

import pickle
#from pylab import *
import synthetic_data

from mlp import MLP
from rd_gp_mlp import rd_mlp
from utils import metrics

def activation(x, name):

    if name.lower() == 'relu':
        relu_x = x.copy()
        relu_x[x <= 0] = 0
        return relu_x
    elif name.lower() == "sigmoid":
        return 1 / (1 + np.exp(-x))

def sigmoid(x):
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

    def gp_fit(self, X, y, gp_lambda=0.1, max_iter=50, lr=0.01, debug=False, tst_X=None, tst_y=None, lasso_model=None, is_sampling=False, is_valid_dim=False, is_step=False):

        assert X.ndim  == 2
        assert X.shape[0] == len(y)

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        tst_X = np.hstack([np.ones((tst_X.shape[0], 1)), tst_X])

        self.N = X.shape[0]
        self.input_dim = X.shape[1]
        self.gp_lambda = gp_lambda

        self.init_lr = lr
        self.lr = np.array([self.init_lr]) #self.init_lr


        self.active_d = [0]
        self.inactive_d = range(1, self.input_dim)

        self._init_gp_param()

        self.is_step = is_step

        mses = []
        tst_mses = []
        vd_mses = []

        is_add = True

        all_X = X.copy()
        all_y = y.copy()

        #Training by iteratively
        while(is_add):
        #max_features = 0
        #while(len(self.active_d) < 50 and max_features < 10000):

            #max_features += 1

            if is_sampling:
                X, X_vd, y, y_vd = train_test_split(all_X, all_y, test_size=0.6)
            else:
                X = X.copy()
                X_vd = X.copy()
                y = y.copy()
                y_vd = y.copy()

            #Train MLP using current active _d
            it_mses = []
            old_v = self.V.copy()
            old_w = self.W.copy()

            #save the gradient norm of each iteration
            grad_w_0 = []

            while True:
            #for tttttt in range(200):
                sgd_shuffle = np.arange(X.shape[0])
                np.random.shuffle(sgd_shuffle)
                batch_size = 10 #X.shape[0]

                it_mses.append(metrics(y, self.gp_predict(X), "logistic"))
                for sgd_i in range(int(math.ceil(X.shape[0] * 1.0 / batch_size))):
                    #print "SGD Tr MSE:%f, Tst MSE:%f" % (metrics(y, self.gp_predict(X), "MSE"), metrics(tst_y, self.gp_predict(tst_X), "MSE"))
                    grad_w = self.gp_update(X[(sgd_i * batch_size):((sgd_i+1) * batch_size), :],y[(sgd_i * batch_size):((sgd_i+1) * batch_size)])
                    self.lr = self.lr * 0.99

                #Save the gradient norm change
                # try:
                #     grad_w_0.append(np.linalg.norm(grad_w,axis=1)[0])
                # except:
                #     grad_w_0.append(np.linalg.norm(grad_w,axis=1))t
                #it_mses.append(metrics(y, self.gp_predict(X), "logistic"))
                if len(it_mses) > 5:
                    if ((it_mses[-5] - it_mses[-1]) < 0):
                        break
                if len(it_mses) > 30:
                    if (((it_mses[-20] - it_mses[-1]) / it_mses[-20]) < 0.0001):
                         break

                #print "Sparsity LW:%f" % np.mean(self.lW == 0)

            #Show the gradient norm
            # figure()
            # plot(grad_w_0, "*-", linewidth=3)
            # show()

            if False:
                close()
                figure()
                title("Current Active Features:%d" % (len(self.active_d)))
                plot(it_mses, "*-", linewidth=3)
                savefig("fig/%d.jpg" % len(self.active_d))

            #mses.append(metrics(y, (self.gp_predict(X)+lasso_model.predict_proba(X[:, 1::])[:,1]) / 2, self.metrics_func))
            #tst_mses.append(metrics(tst_y, (self.gp_predict(tst_X) + lasso_model.predict_proba(tst_X[:, 1::])[:,1]) / 2, self.metrics_func))
            #mses.append(metrics(y, self.gp_predict(X), self.metrics_func))
            #tst_mses.append(metrics(tst_y, self.gp_predict(tst_X), self.metrics_func))
            mses.append(metrics(y, self.gp_predict(X), self.metrics_func))
            tst_mses.append(metrics(tst_y, self.gp_predict(tst_X), self.metrics_func))
            vd_mses.append(metrics(y_vd, self.gp_predict(X_vd), self.metrics_func))
            #########################################
            #Test using concat features.
            #new_tr_x = np.hstack([X, self.apply(X)])
            #new_tst_x = np.hstack([tst_X, self.apply(tst_X)])
            #concat_lr = Lasso(alpha=0.25, fit_intercept=True, normalize=False)
            #concat_lr = LogRCV(penalty="l1", Cs=50, fit_intercept=False, cv=5, n_jobs=-1, refit=True, solver="liblinear", scoring="log_loss")
            #concat_lr.fit(new_tr_x, y)
            #mses.append(metrics(concat_lr.predict_proba(new_tr_x)[:, 1], y, self.metrics_func))
            #tst_mses.append(metrics(concat_lr.predict_proba(new_tst_x)[:, 1], tst_y, self.metrics_func))
            #tst_mses.append(metrics(tst_y, lasso_model.predict(tst_X[:, 1::]) + self.gp_predict(tst_X), self.metrics_func))

            ########################################################################################
            #Test Using MLP
            # nn = Regressor(
            #         layers = [
            #             #Layer("Sigmoid", units=int(np.ceil(len(self.active_d) * 1.0 / 2))),
            #             #Layer("Sigmoid", units=int(np.ceil(len(self.active_d) * 1.0 / 3))),
            #             # Layer("Rectifier", units=50),
            #             Layer("Rectifier", units=30),
            #             # Layer("Rectifier", units=10),
            #             Layer("Linear")
            #         ],
            #         learning_rate=0.0001,
            #         n_iter=50,
            #         regularize="L2",
            #         weight_decay=0.1
            #         )
            nn = Classifier(
                    layers = [
                        #Layer("Sigmoid", units=int(np.ceil(len(self.active_d) * 1.0 / 2))),
                        #Layer("Sigmoid", units=int(np.ceil(len(self.active_d) * 1.0 / 3))),
                        # Layer("Rectifier", units=50),
                        Layer("Rectifier", units=30),
                        Layer("Rectifier", units=15),
                        Layer("Rectifier", units=8),
                        # Layer("Rectifier", units=10),
                        Layer("Softmax")
                    ],
                    learning_rate=0.0001,
                    n_iter=500,
                    regularize="L2",
                    weight_decay=20
                    )
            nn.fit(X[:, self.active_d], y)
            tr_pred = nn.predict_proba(X[:, self.active_d])[:, 1].flatten()
            tst_pred = nn.predict_proba(tst_X[:, self.active_d])[:, 1].flatten() #+ lasso_model.predict(tst_X[:, 1::]).flatten()
            vd_pred = nn.predict_proba(X_vd[:, self.active_d])[:, 1].flatten()
            #mses.append(metrics(y, tr_pred, self.metrics_func))
            #tst_mses.append(metrics(tst_y, tst_pred, self.metrics_func))
            ########################################################################################
            ########################################################################################

            # if len(self.active_d) > 10:
            #     close()
            #     #plot(np.arange(len(y)), y, "r*")
            #     #plot(np.arange(len(tst_y)), nn.predict(tst_X[:, self.active_d]), "b*")
            #     #plot(np.arange(len(y)), self.gp_predict(X), "b*")
            #     #plot(y, self.gp_predict(X),"*")
            #     #plot(y, tr_pred,"*")
            #     tr_pred = lasso_model.predict_proba(X[:, 1::])[:,1]
            #     plot(np.arange(np.sum(y == 1)), tr_pred[y == 1], "r*")
            #     plot(np.arange(np.sum(y == 0)), tr_pred[y == 0], "b*")
            #     legend(["Pos", "Neg"])
            #     title("Training")
            #     show()

            #     close()
            #     #plot(tst_y, tst_pred, "*")
            #     tst_pred = lasso_model.predict_proba(tst_X[:, 1::])[:, 1]
            #     plot(np.arange(np.sum(tst_y == 1)), tst_pred[tst_y == 1], "r*")
            #     plot(np.arange(np.sum(tst_y == 0)), tst_pred[tst_y == 0], "b*")
            #     legend(["Pos", "Neg"])
            #     title("Test")
            #     show()

            #print "=" * 50
            #print "SKNN Iter:%d, Tr Log Loss:%f, Tr %s:%f, Tst %s:%f" % (len(self.active_d), metrics(y, nn.predict_proba(X[:, self.active_d])[:,1].flatten(), "logistic"), self.metrics_func, metrics(y, tr_pred, self.metrics_func), self.metrics_func, metrics(tst_y, tst_pred, self.metrics_func))
            #print "Total Selected:%d, Hidden Selected:%d, Sparsity Tr:%f, Sparsity Tst:%f" % (np.sum(concat_lr.coef_ != 0), np.sum(concat_lr.coef_[len(self.active_d)::] != 0), np.mean(self.apply(X) == 0), np.mean(self.apply(tst_X) == 0))
            #print "Iter:%d, Tr Log Loss:%f, Tr %s:%f, Tst %s:%f" % (len(self.active_d), metrics(y, self.gp_predict(X), "logistic"), self.metrics_func, metrics(y, self.gp_predict(X), self.metrics_func), self.metrics_func, metrics(tst_y, self.gp_predict(tst_X), self.metrics_func))
            #print "Iter:%d, Ensemble Tr %s:%f, Tst %s:%f, SP RELU:%f" % (len(self.active_d), self.metrics_func, mses[-1], self.metrics_func, tst_mses[-1], np.mean(self.hidden_unit == 0))

            # print "Iter:%d, Tr Log Loss:%f, Tr %s:%f, VD %s:%f, Tst %s:%f, SK Tr %s:%f, VD %s: %f, Tst %s:%f" % (len(self.active_d), metrics(y, self.gp_predict(X), "logistic"),
            #             self.metrics_func, metrics(y, self.gp_predict(X), self.metrics_func),
            #             self.metrics_func, metrics(y_vd, self.gp_predict(X_vd), self.metrics_func),
            #             self.metrics_func, metrics(tst_y, self.gp_predict(tst_X), self.metrics_func),
            #             self.metrics_func, metrics(y, tr_pred, self.metrics_func),
            #             self.metrics_func, metrics(y_vd, vd_pred, self.metrics_func),
            #             self.metrics_func, metrics(tst_y, tst_pred, self.metrics_func))
            if False:#(not is_valid_dim) or (not is_sampling):
                print "Iter:%d, Tr Log Loss:%f, Tr %s:%f, VD %s:%f, Tst %s:%f" % (len(self.active_d), metrics(y, self.gp_predict(X), "logistic"),
                            self.metrics_func, metrics(y, self.gp_predict(X), self.metrics_func),
                            self.metrics_func, metrics(y_vd, self.gp_predict(X_vd), self.metrics_func),
                            self.metrics_func, metrics(tst_y, self.gp_predict(tst_X), self.metrics_func))

            if True and is_valid_dim:
                if len(mses) > 2:
                    if vd_mses[-1] <= vd_mses[-2]:
                    #if vd_mses[-1] < vd_mses[-2]:
                        #Do not Incorporate the features
                        self.active_d.pop(-1)
                        self.W = self.W[0:-1, :]
                        self.vW = self.vW[0:-1, :]
                        self.lr = self.lr[0:-1]
                        tst_mses.pop(-1)
                        mses.pop(-1)
                        vd_mses.pop(-1)
                        #print "Error Add Dim"
                    else:
                        print "=" * 50
                        print "Iter:%d, Tr Log Loss:%f, Tr %s:%f, VD %s:%f, Tst %s:%f" % (len(self.active_d), metrics(y, self.gp_predict(X), "logistic"),
                            self.metrics_func, metrics(y, self.gp_predict(X), self.metrics_func),
                            self.metrics_func, metrics(y_vd, self.gp_predict(X_vd), self.metrics_func),
                            self.metrics_func, metrics(tst_y, self.gp_predict(tst_X), self.metrics_func))
                        print "SKNN Iter:%d, Tr Log Loss:%f, Tr %s:%f, VD %s:%f, Tst %s:%f" % (len(self.active_d), metrics(y, nn.predict_proba(X[:, self.active_d])[:,1].flatten(), "logistic"),
                         self.metrics_func, metrics(y, tr_pred, self.metrics_func),
                         self.metrics_func, metrics(y_vd, vd_pred, self.metrics_func),
                         self.metrics_func, metrics(tst_y, tst_pred, self.metrics_func))
                else:
                    print "=" * 50
                    print "Iter:%d, Tr Log Loss:%f, Tr %s:%f, VD %s:%f, Tst %s:%f" % (len(self.active_d), metrics(y, self.gp_predict(X), "logistic"),
                            self.metrics_func, metrics(y, self.gp_predict(X), self.metrics_func),
                            self.metrics_func, metrics(y_vd, self.gp_predict(X_vd), self.metrics_func),
                            self.metrics_func, metrics(tst_y, self.gp_predict(tst_X), self.metrics_func))
                    print "SKNN Iter:%d, Tr Log Loss:%f, Tr %s:%f, VD %s:%f, Tst %s:%f" % (len(self.active_d), metrics(y, nn.predict_proba(X[:, self.active_d])[:,1].flatten(), "logistic"),
                         self.metrics_func, metrics(y, tr_pred, self.metrics_func),
                         self.metrics_func, metrics(y_vd, vd_pred, self.metrics_func),
                         self.metrics_func, metrics(tst_y, tst_pred, self.metrics_func))

            #Add new features
            is_add = self.gp_add_feature(X,y)

    def _init_gp_param(self):

        if self.init_param.lower() == "optimal":
            self.W = np.random.uniform(low=-4*math.sqrt(6.0/(2 * self.hidden_len)), high=4*math.sqrt(6.0/(2 * self.hidden_len)), size=(len(self.active_d), self.hidden_len))
            self.V = np.random.uniform(low=-4*math.sqrt(6.0/(self.hidden_len)), high=4*math.sqrt(6.0/(self.hidden_len)), size=(self.hidden_len+1))
            #self.lW = np.zeros(self.input_dim)
        else:
            self.W = np.random.normal(loc=0, scale=self.init_param, size=(len(self.active_d), self.hidden_len)) # Active Input -> Hidden
            self.V = np.random.normal(loc=0, scale=self.init_param, size=(self.hidden_len+1)) #Hidden to Output
            #self.lW = np.zeros(self.input_dim)

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
        grad_v = - np.dot(err, np.hstack([np.ones((self.hidden_unit.shape[0], 1)), self.hidden_unit])) / X.shape[0] #/ self.N

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

        grad_w = - np.dot(active_X.T, err_I_vj) / X.shape[0] #self.N

        #Regularization
        grad_w = grad_w + self.reg_w * self.W
        grad_v = grad_v + self.reg_v * self.V

        #Momentum
        #self.vW = self.momentum * self.vW + self.lr * grad_w
        self.vV = self.momentum * self.vV + self.init_lr * grad_v
        # print np.tile(self.lr.flatten, (self.hidden_len, 1)).T.shape
        # print grad_w.shape
        # self.momentum * self.vW
        # np.tile(self.lr.flatten, (self.hidden_len, 1)).T * grad_w
        self.vW = self.momentum * self.vW + np.tile(self.lr.flatten(), (self.hidden_len, 1)).T * grad_w

        #Gradient Update
        if self.is_step:
            self.W[1::, :] = self.W[1::, :] - self.vW[1::, :]
        else:
            self.W = self.W - self.vW
        self.V = self.V - self.vV

        # try:
        #     self.lW.fit(X, y-self.nn.predict(X[:, self.active_d]).flatten())
        # except:
        #     self.lW.fit(X, y)

        # self.nn = Regressor(
        #             layers = [
        #                 #Layer("Sigmoid", units=int(np.ceil(len(self.active_d) * 1.0 / 2))),
        #                 #Layer("Sigmoid", units=int(np.ceil(len(self.active_d) * 1.0 / 3))),
        #                 Layer("Rectifier", units=50),
        #                 Layer("Rectifier", units=30),
        #                 # Layer("Rectifier", units=10),
        #                 Layer("Linear")
        #             ],
        #             learning_rate=0.0001,
        #             n_iter=50,
        #             regularize="L2",
        #             dropout_rate=0.1
        #             )
        # self.nn.fit(X[:, self.active_d], y-self.lW.predict(X).flatten())

        # try:
        #     self.lW.fit(X, y-self.nn_predict(X))
        # except:
        #     print self.nn_predict(X)

        # self.lW = self.lW - self.lr * 0.05 / X.shape[0] * (-np.dot(err, X).flatten()) #+ self.reg_v * 10 * self.lW)
        # threshold_lW = np.abs(self.lW) - self.reg_v * 10
        # threshold_lW[threshold_lW <= 0] = 0
        # self.lW = np.sign(self.lW) * threshold_lW

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

        #############################################
        #Drop Redundant Dimensions
        norm_W = np.linalg.norm(self.W, axis=1, ord=self.norm_style).flatten()

        #print "Norm W", norm_W
        #print "LR", self.lr

        drop_id = norm_W < 0.5
        # if np.sum(drop_id) > 0:
        #     print "Drop Dimension", np.arange(len(drop_id))[drop_id]
        #     #self.active_d.pop(-1)
        #     self.active_d = np.array(self.active_d)[np.logical_not(drop_id)].tolist()
        #     self.W = self.W[np.logical_not(drop_id), :]
        #     self.vW = self.vW[np.logical_not(drop_id), :]
        #     self.lr = self.lr[np.logical_not(drop_id)]
        #     print self.active_d

        #############################################

        pred = self.gp_predict(X)
        err = (y - pred)

        #Gradient of W
        v_tile = np.tile(self.V[1::], (X.shape[0], 1)) #N * k
        err_tile = np.tile(err.reshape(len(err), 1), (1, self.hidden_len)) #N * k

        if self.activation_func.lower() == "relu":
            I_kl  = self.hidden_unit.copy() #N * k
            I_kl[self.hidden_unit == 0] = 0.1
            I_kl[self.hidden_unit != 0] = 1
            err_I_vj = err_tile * (I_kl * v_tile)
        elif self.activation_func.lower() == "sigmoid":
            err_I_vj = err_tile * (self.hidden_unit * (1 - self.hidden_unit)) * v_tile

        grad_w = - np.dot(X.T, err_I_vj) / X.shape[0] #/ self.N

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
            self.W = np.vstack([self.W, np.zeros((1, self.hidden_len))])
            #self.W = np.vstack([self.W, np.random.uniform(low=-4*math.sqrt(6.0/(2 * self.hidden_len)), high=4*math.sqrt(6.0/(2 * self.hidden_len)), size=(1, self.hidden_len))])
        else:
            self.W = np.vstack([self.W, np.zeros((1, self.hidden_len))])
            #self.W = np.vstack([self.W, np.random.normal(loc=0, scale=self.init_param, size=(1, self.hidden_len))])
        self.vW = np.vstack([self.vW, np.zeros((1, self.hidden_len))])

        self.lr = np.hstack([self.lr, self.init_lr])

        return True


    def gp_predict(self, X):

        "Forward Neural Network that predict the label"

        # ###########################
        # try:
        #     lW_pred = self.lW.predict(X).flatten()
        # except:
        #     lW_pred = np.array([0] * X.shape[0])

        # try:
        #     nn_pred = self.nn.predict(X[:, self.active_d]).flatten()
        # except:
        #     nn_pred = np.array([0] * X.shape[0])

        # ###########################

    	if len(self.active_d) == 0:
    	    return np.zeros(X.shape[0])

        self.hidden_unit = activation(np.dot(X[:, self.active_d], self.W), self.activation_func) #N * k
        y_pred = np.dot(np.hstack([np.ones((self.hidden_unit.shape[0], 1)), self.hidden_unit]), self.V) #N

        #Cross Level Connection from hidden to Output
        #y_pred = np.dot(X, self.lW)
        # try:
        #     y_pred += self.lW.predict(X).flatten()
        # except:
        #     pass

        #return y_pred
        return sigmoid(y_pred)


    def apply(self, X):
        act = activation(np.dot(X[:, self.active_d], self.W), self.activation_func)

        return act - np.tile(np.mean(act, 0), (act.shape[0], 1))



