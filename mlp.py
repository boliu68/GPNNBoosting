import numpy as np
from  sklearn import datasets
from sklearn.metrics import mean_squared_error as mse
from sklearn.cross_validation import KFold
from sklearn.preprocessing import normalize
from sklearn.linear_model import Lasso as LR
import synthetic_data
# from sklearn.neural_network import MLPRegressor

from pylab import *

def RELU(x):

    relu_x = x.copy()
    relu_x[x <= 0] = 0
    return relu_x

    # return 1 / (1 + np.exp(-x))

def sigmoid(x):

    return 1 / (1 + np.exp(-x))

class MLP:

    def __init__(self, hidden_len=3, activation_func="RELU", init_param=1):

        self.hidden_len = hidden_len
        self.activation_func = activation_func

        self.hidden_unit = np.zeros(hidden_len)
        self.init_param = init_param

    def bp_fit(self, X, y, max_iter=5000, lr=0.01, debug=False):

        assert X.ndim  == 2
        assert X.shape[0] == len(y)

        self.N = X.shape[0]
        self.input_dim = X.shape[1]
        self.lr = lr

        self._init_param()

        mses = []

        #for it in range(max_iter):
        while True:

            self.bp_update(X, y)

            if debug:
                mses.append(mse(self.fpredict(X), y))
                #print "Iter:%d MSE:%f" % (it, mses[-1])

            if mses[-1] < 50:
                break

            print "MSE:%f, Sum H:%f, V:%f" % (mses[-1], np.sum(self.hidden_unit), np.sum(np.abs(self.V)))

	    #self.lr = self.lr / 1.01


	if debug:
            plot(mses,"*-", linewidth=3)
            grid(True)
            show()
            grid(True)


    def _init_param(self):

        self.W = np.random.normal(loc=0, scale=self.init_param, size=(self.input_dim, self.hidden_len)) # Input -> Hidden
        self.V = np.random.normal(loc=0, scale=self.init_param, size=(self.hidden_len)) #Hidden to Output

    def bp_update(self, X, y):

        pred = self.fpredict(X)
        err = (y - pred)

        #Gradient of v
        grad_v = - np.dot(err, self.hidden_unit) / self.N

    	# pred = self.fpredict(X)
     #    err = (y - pred)

        #Gradient of W
        I_kl  = self.hidden_unit.copy() #N * k
        I_kl[self.hidden_unit == 0] = 0
        I_kl[self.hidden_unit != 0] = 1

    	#if True:#np.sum(self.hidden_unit) == 0:
    	#    I_kl[:] = 1

        #print "I KL Average:", np.mean(I_kl)

        v_tile = np.tile(self.V, (self.N, 1)) #N * k
        err_tile = np.tile(err.reshape(len(err), 1), (1, self.hidden_len)) #N * km()

        err_I_vj = err_tile * (I_kl * v_tile)
        #err_I_vj = err_tile * (self.hidden_unit * (1 - self.hidden_unit) * v_tile)

        grad_w = - np.dot(X.T, err_I_vj) / self.N

	    #print "Max V GD:%f, W GD:%f" % (np.max(np.abs(grad_v)), np.max(np.abs(grad_w)))

        self.V = self.V - self.lr * (grad_v + 0 * self.V)
        self.W = self.W - self.lr * (grad_w + 0 * self.W)


    def fpredict(self, X):

        "Forward Neural Network that predict the label"

        self.hidden_unit = RELU(np.dot(X, self.W)) #N * k
        y_pred = np.dot(self.hidden_unit, self.V)

        return y_pred


if __name__ == "__main__":

    print "Hello! Group Lasso Multi Layer Perceptron"
    #X = np.array([[0,1,0], [1,0,1], [0,0,1]])
    #y = [0,1,0]

    #xy = datasets.load_diabetes()
    #X = xy["data"]
    # X = np.hstack([X, np.ones((X.shape[0], 1))])
    # print np.max(X)
    # print np.min(X)
    #y = xy["target"]

    X, y = synthetic_data.generate(100, 0, 20)


    #Add Noise to Data
    noise_d = 500
    noise_mse = []
    for noise_d in [0]:

        noise_X = np.hstack([X, np.random.uniform(np.min(X), np.max(X), size=(X.shape[0], noise_d))])
        noise_mse.append(0)

        kf = KFold(X.shape[0], n_folds=5)

        for tr, tst in kf:
            tr_X = noise_X[tr, :]
            tr_y = y[tr]

            tst_X = noise_X[tst,:]
            tst_y = y[tst]

            cmlp = MLP(hidden_len=50, init_param=0.1)
            cmlp.bp_fit(tr_X,tr_y, lr=0.01, debug=True)

            tr_pred = cmlp.fpredict(tr_X)
            tst_pred = cmlp.fpredict(tst_X)

            plot(tst_pred, tst_y, "*")
            show()

            print "=" * 50
            print "MY MLP TR_MSE:%f, TST MSE:%f" % (mse(tr_pred, tr_y), mse(tst_pred, tst_y))
            noise_mse[-1] += mse(tst_pred, tst_y)

        noise_mse[-1] = noise_mse[-1] / 5


    # plot(noise_mse, "r*-", linewidth=3)
    # legend(["MLP"])
    # xticks(range(5), ["%s" % x for x in range(0, 1001, 200)])
    # ylabel("MSE")
    # show()

