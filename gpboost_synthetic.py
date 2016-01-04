import math
import numpy as np
from  sklearn import datasets
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import Lasso, Ridge, LassoCV
from sknn.mlp import Regressor, Layer

from gpmlp import gp_mlp

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

def load_data(N):
    #X = np.array([[0,1,0], [1,0,1], [0,0,1]])
    #y = [0,1,0]

    #xy = datasets.load_diabetes()
    #X = xy["data"]
    # X = np.hstack([X, np.ones((X.shape[0], 1))])
    #y = xy["target"]

    # X = pickle.load(open("../16k_fea.pkl"))
    # y = pickle.load(open("../16k_gd.pkl"))[:, 31]

    # keep_y = y != -1
    # X = X[keep_y, :]
    # y = y[keep_y]

    # keep_y = y < (np.mean(y) + 5 * np.std(y))
    # X = X[keep_y, :]
    # y = y[keep_y]

    # #Normalize
    # X = X - np.tile(np.mean(X, 0), (X.shape[0], 1))

    X, y, nn = synthetic_data.generate(N=N, hidden_len=100, noisy_d=100, true_dim=100)
    #X = preprocessing.normalize(X, norm="max")

    return X, y, nn

if __name__ == "__main__":

    print "Hello! Group Lasso Multi Layer Perceptron"

    #Add Noise to Data
    noise_mse = []
    noise_mlp_mse = []
    noise_lasso_mse = []
    noise_ridge_mse = []
    rd_mse = []

    tr_lasso_mse = []
    tr_mlp_mse = []

    for sample_size in [100, 500, 1000, 2500, 5000, 10000]:
        X, y, true_nn = load_data(sample_size)

        for noise_d in [0]:#range(0, 1001, 200) + range(2000, 5000, 1000):
            noise_X = np.hstack([X, np.random.uniform(np.min(X), np.max(X), size=(X.shape[0], noise_d))])

            noise_mse.append(0)
            noise_mlp_mse.append(0)
            noise_lasso_mse.append(0)
            noise_ridge_mse.append(0)
            rd_mse.append(0)

            tr_lasso_mse.append(0)
            tr_mlp_mse.append(0)

            kf = KFold(X.shape[0], n_folds=5, shuffle=True)

            for tr, tst in kf:

                #Split Training and Test
                tr_X = noise_X[tr, :]
                tr_y = y[tr]

                tst_X = noise_X[tst,:]
                tst_y = y[tst]

                ###############################################
                #Random Guess
                rd_pred = np.array([np.mean(tr_y)] * len(tst_y))#np.random.uniform(np.min(y), np.max(y), len(tst_y))
                rd_mse[-1] = metrics(rd_pred, tst_y, "MSE")
                tr_rd_mse = metrics(np.array([np.mean(tr_y)] * len(tr_y)), tr_y, "MSE")
                print "=" * 50
                print "Random Tr MSE:%f, Tst MSE:%f" % (tr_rd_mse, rd_mse[-1])

                ###############################################
                # #Training Lasso
                ls = Lasso(alpha=0.25, normalize=False, fit_intercept=True)
                ls.fit(tr_X[:,0:100], tr_y)

                #print "Lasso Non Zero:%d,%f" % (np.sum(ls.coef_ != 0), np.sum(ls.coef_))

                tr_pred = ls.predict(tr_X[:,0:100])
                tst_pred = ls.predict(tst_X[:,0:100])

                #plot(tst_y, tst_pred, "*")
                #show()

                print "Noise Level:%d, NNZ:%d, Lasso TR_MSE:%f, TST MSE:%f, Correct Selection:%f" % (
                    noise_d, np.sum(ls.coef_!=0), metrics(tr_pred, tr_y, "MSE"),
                    metrics(tst_pred, tst_y, "MSE"),
                    np.mean(ls.coef_[0:100] != 0))

                noise_lasso_mse[-1] += metrics(tst_pred, tst_y, "MSE")
                tr_lasso_mse[-1] += metrics(tr_pred, tr_y, "MSE")

                ###############################################
                #Training Multi Layer Perceptron

                print "Optimal NN Tr:%f, Tst:%f" % (metrics(true_nn.predict(tr_X[:, 0:100]), tr_y, "MSE"), metrics(true_nn.predict(tst_X[:, 0:100]), tst_y, "MSE"))

                #for regu_param in [0, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 50, 500]:
                best_tst_mse = 1000000
                best_tr_mse = 10000000
                for hidden_unit_len  in [50, 100, 200, 300, 400, 500]:
                    nn = Regressor(
                        layers = [
                            Layer("Rectifier", units=hidden_unit_len),
                            #Layer("Rectifier", units=50),
                            Layer("Linear")
                        ],
                        learning_rate=0.00001,
                        n_iter=500,
                        regularize="L1",
                        weight_decay=0.5
                        )
                    #nn.fit(np.hstack([np.ones((tr_X.shape[0], 1)), tr_X[:, ls.coef_!=0]]), tr_y)#-ls.predict(tr_X))
                    #tr_pred = nn.predict(np.hstack([np.ones((tr_X.shape[0], 1)), tr_X[:, ls.coef_!=0]])).flatten() #+ ls.predict(tr_X).flatten()
                    #tst_pred = nn.predict(np.hstack([np.ones((tst_X.shape[0], 1)), tst_X[:, ls.coef_!=0]])).flatten() #+ ls.predict(tst_X).flatten()

                    nn.fit(tr_X[:, 0:100], tr_y)
                    tr_pred = nn.predict(tr_X[:, 0:100])
                    tst_pred = nn.predict(tst_X[:, 0:100])
                    #plot(tst_y, tst_pred, "*")
                    #show()
                    if metrics(tst_pred, tst_y, "MSE") < best_tst_mse:
                        best_tst_mse = metrics(tst_pred, tst_y, "MSE")
                        best_tr_mse = metrics(tr_pred, tr_y, "MSE")

                    print "Len Unit:%d, MLP TR_MSE:%f, TST MSE:%f" % (hidden_unit_len, metrics(tr_pred, tr_y,"MSE"), metrics(tst_pred, tst_y, "MSE"))

                noise_mlp_mse[-1] += best_tst_mse
                tr_mlp_mse[-1] += best_tr_mse

            noise_lasso_mse[-1] /= 5
            noise_mlp_mse[-1] /= 5
            tr_lasso_mse[-1] /= 5
            tr_mlp_mse[-1] /= 5

    pickle.dump([noise_mlp_mse, noise_lasso_mse, tr_mlp_mse, tr_lasso_mse], open("SampleSize_Perf_MLPLS.pkl", "w"))

            ###############################################
            #Training Group Lasso Boosting
            # gpmlp = gp_mlp(hidden_len=100, reg_v=0.0005, reg_w=0, momentum=0, init_param='optimal', activation_func="RELU", metrics_func="MSE")
            # gpmlp.gp_fit(tr_X,tr_y, gp_lambda=0, lr=0.001, max_iter=600, debug=False, tst_X=tst_X, tst_y=tst_y, lasso_model=ls)

            # tr_pred = gpmlp.gp_predict(tr_X)
            # tst_pred = gpmlp.gp_predict(tst_X)

            # print "Noise Level:%d, GP #Selection:%d, GP MLP TR_MSE:%f, TST MSE:%f" % (noise_d, len(gpmlp.active_d), metrics(tr_pred, tr_y,"MSE"), metrics(tst_pred, tst_y, "MSE"))
            # noise_mse[-1] += metrics(tst_pred, tst_y, "MSE")

            ###############################################


            # gpmlp = rd_mlp(hidden_len=10, init_param=0.001)
            # gpmlp.gp_fit(tr_X,tr_y, gp_lambda=0, lr=0.01, max_iter=600, debug=False, tst_X=tst_X, tst_y=tst_y)

            # tr_pred = gpmlp.gp_predict(tr_X)
            # tst_pred = gpmlp.gp_predict(tst_X)

            # print "Noise Level:%d, GP #Selection:%d, GP MLP TR_MSE:%f, TST MSE:%f" % (noise_d, len(gpmlp.active_d), mse(tr_pred, tr_y), mse(tst_pred, tst_y))
            # noise_mse[-1] += mse(tst_pred, tst_y)

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

    #     noise_mse[-1] = noise_mse[-1] / 5
    #     noise_mlp_mse[-1] = noise_mlp_mse[-1] / 5
    #     noise_lasso_mse[-1] = noise_lasso_mse[-1] / 5
    #     noise_ridge_mse[-1] = noise_ridge_mse[-1] / 5

    # print "GP Boost:%f, MLP:%f, Lasso:%f, Ridge:%f" % (noise_mse[-1], noise_mlp_mse[-1], noise_lasso_mse[-1], noise_ridge_mse[-1])
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
