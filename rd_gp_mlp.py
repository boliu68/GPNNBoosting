import math
import numpy as np
from  sklearn import datasets
from sklearn.metrics import mean_squared_error as mse
from sklearn.cross_validation import KFold
from sklearn import preprocessing
from sklearn.linear_model import Lasso, Ridge, LassoCV
# from sklearn.neural_network import MLPRegressor

import synthetic_data
from pylab import *
import pickle

from mlp import MLP

def auc(pred, y):

    n = 0
    correct = 0

    for i in range(len(y)):
       for j in range(i+1, len(y)):
            if (pred[i] - pred[j]) * (y[i] - y[j]) > 0:
                correct += 1

            n += 1

    return correct * 1.0 / n

def RELU(x):

    # relu_x = x.copy()
    # relu_x[x <= 0] = 0
    # return relu_x

    return 1 / (1 + np.exp(-x))

def sigmoid(x):

    return 1 / (1 + np.exp(-x))

class rd_mlp:

    def __init__(self, hidden_len=3, activation_func="RELU", init_param=1):

        self.hidden_len = hidden_len
        self.activation_func = activation_func

        self.hidden_unit = np.zeros(hidden_len)
        self.init_param = init_param

        self.active_d = []
        self.inactive_d = []

    def gp_fit(self, X, y, gp_lambda=0.1, max_iter=50, lr=0.01, debug=False, tst_X=None, tst_y=None):

        assert X.ndim  == 2
        assert X.shape[0] == len(y)

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        tst_X = np.hstack([np.ones((tst_X.shape[0], 1)), tst_X])

        self.N = X.shape[0]
        self.input_dim = X.shape[1]
        self.init_lr = lr
        self.lr = lr
        self.lr = np.array([lr])
        self.gp_lambda = gp_lambda

        self.active_d = [0]
        self.inactive_d = range(1, self.input_dim)

        self._init_gp_param()

        mses = []
        is_add = True

        #Training
        while(is_add):

            #Train MLP using current active _d
            it_mses = []
            old_v = self.V.copy()
            old_w = self.W.copy()


            #self.vW = np.zeros(self.W.shape)
            #self.vV = np.zeros(self.V.shape)

            #for it in range(max_iter):
            #if len(self.active_d) == 20:
            grad_w_0 = []
            while True:
                sgd_shuffle = np.arange(X.shape[0])
                np.random.shuffle(sgd_shuffle)
                batch_size = X.shape[0]

                #for sgd_one in sgd_shuffle:
                for sgd_i in range(int(math.ceil(X.shape[0] / batch_size))):
                    #self.gp_update(X[sgd_one:sgd_one+1, :],y[sgd_one:sgd_one+1])
                    grad_w = self.gp_update(X[(sgd_i * batch_size):((sgd_i+1) * batch_size), :],y[(sgd_i * batch_size):((sgd_i+1) * batch_size)])

                # try:
                #     grad_w_0.append(np.linalg.norm(grad_w,axis=1)[0])
                # except:
                #     grad_w_0.append(np.linalg.norm(grad_w,axis=1))

                it_mses.append(mse(self.gp_predict(X), y))
                if len(it_mses) > 5:
                    if ((it_mses[-5] - it_mses[-1]) < 0):
                        break
                if len(it_mses) > 30:
                    if (((it_mses[-20] - it_mses[-1]) / it_mses[-20]) < 0.00001):
                         break

            # figure()
            # plot(grad_w_0, "*-", linewidth=3)
            # show()

            if True:#len(self.active_d) % 5 == 1:
                figure()
                title("Current Active Features:%d" % (len(self.active_d)))
                plot(it_mses, "*-", linewidth=3)
                #show()
                savefig("fig/%d.jpg" % len(self.active_d))

            mses.append(mse(self.gp_predict(X), y))
            print "Iter:%d, Tr MSE:%f, Tst MSE:%f, V[0]:%f, SP RELU:%f" % (len(self.active_d), mses[-1], mse(tst_y, self.gp_predict(tst_X)), self.V[0], np.mean(self.hidden_unit == 0))

            # if len(mses) > 2:
            #     if mses[-1] > mses[-2] or it_mses[-1] > it_mses[0]:
            #         #Do not Incorporate the features
            #         self.active_d.pop(-1)
            #         self.W = self.W[0:-1, :]
            #         self.vW = self.vW[:-1, :]
            #         mses.pop(-1)
            #         print "Error Add Dim"
            #     else:
            #         print "Iter:%d, Tr MSE:%f, Tst MSE:%f, V[0]:%f" % (len(self.active_d), mses[-1], mse(tst_y, self.gp_predict(tst_X)), self.V[0])
            #         if True:#len(self.active_d) % 5 == 1:
            #             figure()
            #             title("Current Active Features:%d" % (len(self.active_d)))
            #             plot(it_mses, "*-", linewidth=3)
            #             #show()
            #             savefig("fig/%d.jpg" % len(self.active_d))

            #Add new features
            is_add = self.gp_add_feature(X,y)
            #self.lr = self.lr / 1.1
            self.lr = np.hstack([self.lr, [self.init_lr]])
            #print "New Added Feature:%d" % self.active_d[-1]
            # if debug:

        if debug:
            # plot(mses,"*-", linewidth=3)
            # grid(True)
            # #show()
            pass


    def _init_gp_param(self):
        self.W = np.random.normal(loc=0, scale=self.init_param, size=(len(self.active_d), self.hidden_len)) # Active Input -> Hidden
        self.V = np.random.normal(loc=0, scale=self.init_param, size=(self.hidden_len+1)) #Hidden to Output

        # self.W = np.zeros((len(self.active_d), self.hidden_len))
        # self.V = np.zeros(self.hidden_len+1)

        self.vW = np.zeros(self.W.shape)
        self.vV = np.zeros(self.V.shape)

    def gp_update(self, X, y):

        if len(self.active_d) == 0:
            return

        pred = self.gp_predict(X)
        err = (y - pred)
        active_X = X[:, self.active_d]

        #Gradient of v
        #grad_v = - np.dot(err, self.hidden_unit) / self.N
        grad_v = - np.dot(err, np.hstack([np.ones((self.hidden_unit.shape[0], 1)), self.hidden_unit])) / self.N

        #Gradient of W
        I_kl  = self.hidden_unit.copy() #N * k
        I_kl[self.hidden_unit == 0] = 0.1
        I_kl[self.hidden_unit != 0] = 1

        #v_tile = np.tile(self.V, (self.N, 1)) #N * k
        v_tile = np.tile(self.V[1::], (X.shape[0], 1))
        err_tile = np.tile(err.reshape(len(err), 1), (1, self.hidden_len)) #N * k

        #err_I_vj = err_tile * (I_kl * v_tile)
        err_I_vj = err_tile * (self.hidden_unit * (1 - self.hidden_unit)) * v_tile

        grad_w = - np.dot(active_X.T, err_I_vj) / self.N

        #self.V = self.V - self.lr * (grad_v + 0 * self.V)
        #self.W = self.W - self.lr * (grad_w + 1 * self.W)

        #Momentum

        lr = np.tile(self.lr.reshape(len(self.lr), 1), (1, grad_w.shape[1]))

        self.vW = 0.5 * self.vW + lr * (grad_w + 0.1 * self.W)
        self.vV = 0.5 * self.vV + self.init_lr * (grad_v + 0.1 * self.V)

        self.W = self.W - self.vW
        self.V = self.V - self.vV

        #Lasso
        # threshold_V = (np.abs(self.V) - self.lr * 0.1)
        # threshold_V[threshold_V < 0] = 0
        # self.V = np.sign(self.V) * threshold_V

        # threshold_W = (np.abs(self.W) - self.lr * 0.1)
        # threshold_W[threshold_W < 0] = 0
        # self.W = np.sign(self.W) * threshold_W

        return grad_w

    def gp_add_feature(self, X, y):

        if len(self.inactive_d) == 0:
            return False

        pred = self.gp_predict(X)
        err = (y - pred)

        #Gradient of W
        I_kl  = self.hidden_unit.copy() #N * k
        I_kl[self.hidden_unit == 0] = 0.1
        I_kl[self.hidden_unit != 0] = 1

        #print "I kl Mean:%f" % np.mean(I_kl == 0)
        # if len(self.active_d) == 0:
        #     I_kl[:] = 1
            #print "I KL Average:", np.mean(I_kl)

        v_tile = np.tile(self.V[1::], (self.N, 1)) #N * k
        err_tile = np.tile(err.reshape(len(err), 1), (1, self.hidden_len)) #N * k
        #err_I_vj = err_tile * (I_kl * v_tile)
        err_I_vj = err_tile * (self.hidden_unit * (1 - self.hidden_unit)) * v_tile

        grad_w = - np.dot(X.T, err_I_vj) / self.N

        inactive_norm_grad = np.linalg.norm(grad_w, axis=1)[self.inactive_d]
        #inactive_id = np.argmax(inactive_norm_grad)
        inactive_id = np.random.randint(0, len(inactive_norm_grad))

        # plot(inactive_norm_grad * self.lr, "*-", linewidth=3)
        # title("Distributed of Norm of Gradient")
        # show()
        # print "Largest Gradient:%f" % (inactive_norm_grad[inactive_id] * self.lr)

        if inactive_norm_grad[inactive_id] * self.init_lr <= self.gp_lambda:
            #End Training
            return False

        #print "Max Gradient L2 Norm %f " % inactive_norm_grad[inactive_id]

        self.active_d.append(self.inactive_d[inactive_id])
        self.inactive_d.pop(inactive_id)


        self.W = np.vstack([self.W, np.random.normal(loc=0, scale=self.init_param, size=(1, self.hidden_len))])
        self.vW = np.vstack([self.vW, np.zeros((1, self.hidden_len))])

        return True


    def gp_predict(self, X):

        "Forward Neural Network that predict the label"

        if len(self.active_d) == 0:
            return np.zeros(X.shape[0])

        self.hidden_unit = RELU(np.dot(X[:, self.active_d], self.W)) #N * k
        #self.hidden_unit = np.hstack([np.ones((self.hidden_unit.shape[0], 1)), self.hidden_unit]) #N * k + 1
        #y_pred = np.dot(self.hidden_unit, self.V)

        y_pred = np.dot(np.hstack([np.ones((self.hidden_unit.shape[0], 1)), self.hidden_unit]), self.V) #N

        return y_pred


if __name__ == "__main__":

    print "Hello! Group Lasso Multi Layer Perceptron"
    #X = np.array([[0,1,0], [1,0,1], [0,0,1]])
    #y = [0,1,0]

    #xy = datasets.load_diabetes()
    #X = xy["data"]
    # X = np.hstack([X, np.ones((X.shape[0], 1))])
    #y = xy["target"]

    X = pickle.load(open("../16k_fea.pkl"))
    y = pickle.load(open("../16k_gd.pkl"))[:, 27]

    keep_y = y != -1
    X = X[keep_y, :]
    y = y[keep_y]

    #X, y = synthetic_data.generate(100, 0, 20)

    #X = preprocessing.normalize(X, norm="max")

    #Add Noise to Data
    noise_mse = []
    noise_mlp_mse = []
    noise_lasso_mse = []
    noise_ridge_mse = []
    rd_mse = []

    for noise_d in [0]:#range(0, 1001, 200) + range(2000, 5000, 1000):
        noise_X = np.hstack([X, np.random.uniform(np.min(X), np.max(X), size=(X.shape[0], noise_d))])

        noise_mse.append(0)
        noise_mlp_mse.append(0)
        noise_lasso_mse.append(0)
        noise_ridge_mse.append(0)
        rd_mse.append(0)

        kf = KFold(X.shape[0], n_folds=5, shuffle=True)

        for tr, tst in kf:

            #Split Training and Test
            tr_X = noise_X[tr, :]
            tr_y = y[tr]

            tst_X = noise_X[tst,:]
            tst_y = y[tst]

            #Random Guess
            rd_pred = np.array([np.mean(tr_y)] * len(tst_y))#np.random.uniform(np.min(y), np.max(y), len(tst_y))
            rd_mse[-1] = mse(rd_pred, tst_y)
            tr_rd_mse = mse(np.array([np.mean(tr_y)] * len(tr_y)), tr_y)
            print "=" * 50
            print "Random Tr MSE:%f, Tst MSE:%f" % (tr_rd_mse, rd_mse[-1])


            # #Training Lasso
            ls = Lasso(alpha=0.25, normalize=False, fit_intercept=True)
            ls.fit(tr_X, tr_y)

            #print "Lasso Non Zero:%d,%f" % (np.sum(ls.coef_ != 0), np.sum(ls.coef_))

            tr_pred = ls.predict(tr_X)
            tst_pred = ls.predict(tst_X)

            print "Noise Level:%d, NNZ:%d, Lasso TR_MSE:%f, TST MSE:%f" % (noise_d, np.sum(ls.coef_!=0), mse(tr_pred, tr_y), mse(tst_pred, tst_y))
            noise_lasso_mse[-1] += mse(tst_pred, tst_y)


            #Training Group Lasso Boosting
            gpmlp = gp_mlp(hidden_len=10, init_param=0.001)
            gpmlp.gp_fit(tr_X,tr_y, gp_lambda=0, lr=0.01, max_iter=600, debug=False, tst_X=tst_X, tst_y=tst_y)

            tr_pred = gpmlp.gp_predict(tr_X)
            tst_pred = gpmlp.gp_predict(tst_X)

            print "Noise Level:%d, GP #Selection:%d, GP MLP TR_MSE:%f, TST MSE:%f" % (noise_d, len(gpmlp.active_d), mse(tr_pred, tr_y), mse(tst_pred, tst_y))
            noise_mse[-1] += mse(tst_pred, tst_y)

            # figure()
            # plot(tst_pred, tst_y, "*")
            # title("GP Boost Result")
            # show()

            #print "\n".join(["Pred:%f, Y:%f" % (tst_pred[i], tst_y[i]) for i in range(20)])

            # #Training classical Multi Layer Percptron
            # cmlp = MLP(hidden_len=200, init_param=0.1)
            # cmlp.bp_fit(tr_X,tr_y, lr=0.01, debug=True,max_iter=1000)

            # tr_pred = cmlp.fpredict(tr_X)
            # tst_pred = cmlp.fpredict(tst_X)

            # print "Noise Level:%d, MY MLP TR_MSE:%f, TST MSE:%f" % (noise_d, mse(tr_pred, tr_y), mse(tst_pred, tst_y))
            # noise_mlp_mse[-1] += mse(tst_pred, tst_y)
#
#       continue


#       #Train MLP with selected features
#            ls_mlp = MLP(hidden_len=80, init_param=0.1)
#       ls_mlp.bp_fit(tr_X[:,ls.coef_!=0],tr_y, lr=0.01,max_iter=300, debug=False)
#
#       tr_pred = ls_mlp.fpredict(tr_X[:, ls.coef_!=0])
#       tst_pred = ls_mlp.fpredict(tst_X[:, ls.coef_!=0])
#
#            print "Noise Level:%d, Lasso MLP TR_MSE:%f, TST MSE:%f" % (noise_d, mse(tr_pred, tr_y), mse(tst_pred, tst_y))
#
#            # #Training Ridge Regression
#            lr = Ridge(alpha=0.1)
#       lr.fit(tr_X[:,ls.coef_!=0], tr_y)
#       tr_pred = lr.predict(tr_X[:, ls.coef_!=0])
#       tst_pred = lr.predict(tst_X[:, ls.coef_!=0])
#
#            print "Noise Level:%d, Ridge TR_MSE:%f, TST MSE:%f" % (noise_d, mse(tr_pred, tr_y), mse(tst_pred, tst_y))
#            noise_ridge_mse[-1] += mse(tst_pred, tst_y)

        noise_mse[-1] = noise_mse[-1] / 5
        noise_mlp_mse[-1] = noise_mlp_mse[-1] / 5
        noise_lasso_mse[-1] = noise_lasso_mse[-1] / 5
        noise_ridge_mse[-1] = noise_ridge_mse[-1] / 5

    print "GP Boost:%f, MLP:%f, Lasso:%f, Ridge:%f" % (noise_mse[-1], noise_mlp_mse[-1], noise_lasso_mse[-1], noise_ridge_mse[-1])
    # pickle.dump([noise_mse, noise_mlp_mse, noise_lasso_mse, noise_ridge_mse], open("mlp_gp_lslr_mse.pkl", "w"))

    # plot(noise_mlp_mse, "b*-", linewidth=3)
    # plot(noise_mse, "r*-", linewidth=3)
    # plot(noise_lasso_mse, "y*-", linewidth=3)
    # plot(noise_ridge_mse, "g*-", linewidth=3)

    # print noise_mlp_mse
    # print noise_mse
    # print noise_lasso_mse
    # print noise_ridge_mse

    # legend(["MLP", "GP MLP", "Lasso", "Ridge"], loc=1)
    # xticks(range(9), ["%s" % x for x in (range(0, 1001, 200) + range(2000, 5000, 1000))])
    # ylabel("MSE")
    # xlabel("The number of noisy dimensions")
    # grid(True)
    # show()
