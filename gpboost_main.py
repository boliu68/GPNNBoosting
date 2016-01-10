import math
import numpy as np
from scipy import io
from  sklearn import datasets
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import Lasso, Ridge, LassoCV, LogisticRegression as LogR, LogisticRegressionCV as LogRCV
from sknn.mlp import Regressor, Layer

from gpmlp import gp_mlp

import pickle
#from pylab import *
import synthetic_data

from mlp import MLP
from rd_gp_mlp import rd_mlp
from utils import metrics
from sklearn import metrics as sk_metrics

from sklearn.ensemble import GradientBoostingRegressor as GBR, GradientBoostingClassifier as GBC
from sklearn.svm import SVR, SVC

#from ae_model import ae_lr

from mx_GPBoost import mx_gp_mlp

def load_data():
    #X = np.array([[0,1,0], [1,0,1], [0,0,1]])
    #y = [0,1,0]

    # xy = datasets.load_digits()
    # X = xy["data"]
    # #X = np.hstack([X, np.ones((X.shape[0], 1))])
    # y = xy["target"]

    # binary_id = np.logical_or(y == 0, y == 1)
    # X = X[binary_id, :]
    # y = y[binary_id]

    # X = X / 255

    # nn = None

    # X = pickle.load(open("../16k_fea.pkl"))
    # y = pickle.load(open("../16k_gd.pkl"))[:, 31]

    # keep_y = y != -1
    # X = X[keep_y, :]
    # y = y[keep_y]

    # keep_y = y < (np.mean(y) + 5 * np.std(y))
    # X = X[keep_y, :]
    # y = y[keep_y]

    # nn = None

    #Normalize
    #X = X - np.tile(np.mean(X, 0), (X.shape[0], 1))

    #X, y, nn = synthetic_data.generate(N=500, hidden_len=100, noisy_d=100, true_dim=100)
    #X = preprocessing.normalize(X, norm="max")


    ####################################
    ########Load SMK Can Dataset########
    #data_dict = io.loadmat("/Users/boliu/Documents/Dropbox/Research/High Dimension/Data/SMK-CAN-187.mat")
    data_dict = io.loadmat("SMK-CAN-187.mat")
    #data_dict = io.loadmat("/Users/boliu/Documents/Dropbox/Research/High Dimension/Data/GLI-85.mat")
    X = data_dict["X"]
    y = data_dict["Y"].flatten()
    X.dtype=np.float64
    y[y==2] = 0
    nn = None
    X = X - np.tile(np.mean(X, 0), (X.shape[0], 1))
    #X = np.random.rand(X.shape[0], X.shape[1])
    #X = (X - np.min(X)) / (np.max(X) - np.min(X))
    ####################################
    return X, y, nn

if __name__ == "__main__":

    print "Hello! Group Lasso Multi Layer Perceptron"

    X, y, true_nn = load_data()

    #Add Noise to Data
    noise_mse = []
    noise_mlp_mse = []
    noise_lasso_mse = []
    noise_ridge_mse = []
    rd_mse = []

    metrics_style = "auc"

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

            ###############################################
            #Random Guess
            rd_pred = np.array([np.mean(tr_y)] * len(tst_y))#np.random.uniform(np.min(y), np.max(y), len(tst_y))
            rd_mse[-1] = metrics(tst_y, rd_pred, metrics_style)
            tr_rd_mse = metrics(tr_y, np.array([np.mean(tr_y)] * len(tr_y)), metrics_style)
            print "=" * 50
            print "Random Tr %s:%f, Tst %s:%f" % (metrics_style, tr_rd_mse, metrics_style,rd_mse[-1])

            ###############################################
            #Training Lasso
#            ls = Lasso(alpha=0.25, normalize=False, fit_intercept=True)
            ls = LogRCV(penalty="l1", Cs=50, fit_intercept=True, cv=5, n_jobs=-1, refit=True, solver="liblinear", scoring="log_loss")
            ls.fit(tr_X, tr_y)

            #print "Lasso Non Zero:%d,%f" % (np.sum(ls.coef_ != 0), np.sum(ls.coef_))

            tr_pred = ls.predict_proba(tr_X)[:, 1]
            tst_pred = ls.predict_proba(tst_X)[:, 1]

            #plot(tst_y, tst_pred, "*")
            #show()

            print "Lasso, NNZ:%d, Tr Log Loss:%f, Tst Log Loss:%f, Lasso TR %s:%f, TST %s:%f" % (
                np.sum(ls.coef_!=0),
                metrics(tr_y, tr_pred, "logistic"),
                metrics(tst_y, tst_pred, "logistic"),
                metrics_style, metrics(tr_y, tr_pred, metrics_style),
                metrics_style, metrics(tst_y, tst_pred, metrics_style))

            noise_lasso_mse[-1] += metrics(tst_y, tst_pred, metrics_style)


            ###############################################
            ################Fit AE LR######################
            #self, activation="Sigmoid", type="denoising", units
            # dae = ae_lr(units=tr_X.shape[1])
            # dae.fit(tr_X, tr_y)

            # tr_pred = ls.predict_proba(tr_X)[:, 1]
            # tst_pred = ls.predict_proba(tst_X)[:, 1]

            # print "DAE, NNZ:%d, Lasso TR %s:%f, TST %s:%f" % (
            #     np.sum(dae.ls.coef_!=0),
            #     metrics_style, metrics(tr_y, tr_pred, metrics_style),
            #     metrics_style, metrics(tst_y, tst_pred, metrics_style))

            # exit()

            ###############################################
            #Use different model to fit the residuals
            # gbr = GBC(loss='deviance')
            # gbr.fit(tr_X, tr_y)
            # print "GBDT Tr %s:%f, Tst %s:%f" % (metrics_style, metrics(tr_y, gbr.predict_proba(tr_X)[:, 1], metrics_style), metrics_style, metrics(tst_y, gbr.predict_proba(tst_X)[:, 1], metrics_style))

            # svr = SVC(probability=True)
            # svr.fit(tr_X, tr_y)
            # print "SVR Tr %s:%f, Tst %s:%f" % (metrics_style, metrics(tr_y, svr.predict_proba(tr_X)[:, 1], metrics_style), metrics_style, metrics(tst_y, svr.predict_proba(tst_X)[:,1], metrics_style))
            ###############################################
            ###########Fit MX GP Boosting #################
            mx_mlp = mx_gp_mlp(hidden_len=[20, 2], reg=0.05, momentum=0.9, init_param=1, activation_func="RELU", metrics_func=metrics_style)
            mx_mlp.gp_fit(tr_X,tr_y, gp_lambda=0, max_features=np.sum(ls.coef_[0]!=0), lr=0.05, max_iter=1000, debug=False, tst_X=tst_X, tst_y=tst_y)

            print "#" * 50
            print "Lasso Select Features:"
            ls_select = np.arange(X.shape[1])[ls.coef_[0] != 0]
            print ls_select
            print "Boost Select Features:"
            print mx_mlp.active_d
            input_param = mx_mlp.input_param.asnumpy()[:, mx_mlp.active_d]
            print np.linalg.norm(input_param, axis=0)
            print "Overlap of two methods:%f" % (len(set(ls_select) & set(mx_mlp.active_d)) * 1.0 / len(ls_select))
            print "#" * 50

            ###############################################
            #Training Multi Layer Perceptron

            # try:
            #     print "Optimal NN Tr:%f, Tst:%f" % (metrics(true_nn.predict(tr_X[:, 0:100]), tr_y, "MSE"), metrics(true_nn.predict(tst_X[:, 0:100]), tst_y, "MSE"))
            # except:
            #     pass

            # #for regu_param in [0, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 50, 500]:
            # for hidden_unit_len  in [50, 100, 200, 300, 400, 500]:
            #     nn = Regressor(
            #         layers = [
            #             Layer("Rectifier", units=hidden_unit_len),
            #             #Layer("Rectifier", units=50),
            #             Layer("Linear")
            #         ],
            #         learning_rate=0.00001,
            #         n_iter=500,
            #         regularize="L1",
            #         weight_decay=0.5
            #         )
            #     #nn.fit(np.hstack([np.ones((tr_X.shape[0], 1)), tr_X[:, ls.coef_!=0]]), tr_y)#-ls.predict(tr_X))
            #     #tr_pred = nn.predict(np.hstack([np.ones((tr_X.shape[0], 1)), tr_X[:, ls.coef_!=0]])).flatten() #+ ls.predict(tr_X).flatten()
            #     #tst_pred = nn.predict(np.hstack([np.ones((tst_X.shape[0], 1)), tst_X[:, ls.coef_!=0]])).flatten() #+ ls.predict(tst_X).flatten()

            #     nn.fit(tr_X[:, 0:100], tr_y)
            #     tr_pred = nn.predict(tr_X[:, 0:100])
            #     tst_pred = nn.predict(tst_X[:, 0:100])

            #     #plot(tst_y, tst_pred, "*")
            #     #show()

            #     print "Len Unit:%d, MLP TR_MSE:%f, TST MSE:%f" % (hidden_unit_len, metrics(tr_pred, tr_y,"MSE"), metrics(tst_pred, tst_y, "MSE"))
            # exit()
            ###############################################
            #Training Group Lasso Boosting
            # gpmlp = gp_mlp(hidden_len=10, reg_v=0.5, reg_w=0.5, momentum=0.9, init_param='optimal', activation_func="RELU", metrics_func=metrics_style)
            # gpmlp.gp_fit(tr_X,tr_y, gp_lambda=0, lr=0.001, max_iter=600, debug=False, tst_X=tst_X, tst_y=tst_y, lasso_model=ls, is_sampling=False, is_valid_dim=False, is_step=False)

            # print "=" * 50
            # gpmlp = gp_mlp(hidden_len=10, reg_v=0.5, reg_w=0.5, momentum=0.9, init_param='optimal', activation_func="RELU", metrics_func=metrics_style)
            # gpmlp.gp_fit(tr_X,tr_y, gp_lambda=0, lr=0.001, max_iter=600, debug=False, tst_X=tst_X, tst_y=tst_y, lasso_model=ls, is_sampling=True, is_valid_dim=False, is_step=False)

            # print "=" * 50
            # gpmlp = gp_mlp(hidden_len=10, reg_v=0.5, reg_w=0.5, momentum=0.9, init_param='optimal', activation_func="RELU", metrics_func=metrics_style)
            # gpmlp.gp_fit(tr_X,tr_y, gp_lambda=0, lr=0.001, max_iter=600, debug=False, tst_X=tst_X, tst_y=tst_y, lasso_model=ls, is_sampling=False, is_valid_dim=True, is_step=False)

            # print "=" * 50
            # gpmlp = gp_mlp(hidden_len=10, reg_v=0.5, reg_w=0.5, momentum=0.9, init_param='optimal', activation_func="Sigmoid", metrics_func=metrics_style)
            # gpmlp.gp_fit(tr_X,tr_y, gp_lambda=0, lr=0.001, max_iter=600, debug=False, tst_X=tst_X, tst_y=tst_y, lasso_model=ls, is_sampling=False, is_valid_dim=False, is_step=False)

            # print "=" * 50
            # gpmlp = gp_mlp(hidden_len=10, reg_v=0.5, reg_w=0.5, momentum=0.9, init_param='optimal', activation_func="RELU", metrics_func=metrics_style)
            # gpmlp.gp_fit(tr_X,tr_y, gp_lambda=0, lr=0.001, max_iter=600, debug=False, tst_X=tst_X, tst_y=tst_y, lasso_model=ls, is_sampling=False, is_valid_dim=False, is_step=True)

            # print "=" * 50
            # gpmlp = gp_mlp(hidden_len=10, reg_v=0.5, reg_w=0.5, momentum=0.9, init_param='optimal', activation_func="RELU", metrics_func=metrics_style)
            # gpmlp.gp_fit(tr_X,tr_y, gp_lambda=0, lr=0.001, max_iter=600, debug=False, tst_X=tst_X, tst_y=tst_y, lasso_model=ls, is_sampling=True, is_valid_dim=False, is_step=True)

            # print "=" * 50
            # gpmlp = gp_mlp(hidden_len=10, reg_v=0.5, reg_w=0.5, momentum=0.9, init_param='optimal', activation_func="RELU", metrics_func=metrics_style)
            # gpmlp.gp_fit(tr_X,tr_y, gp_lambda=0, lr=0.001, max_iter=600, debug=False, tst_X=tst_X, tst_y=tst_y, lasso_model=ls, is_sampling=True, is_valid_dim=True, is_step=True)

            exit()
            #tr_pred = gpmlp.gp_predict(tr_X)
            #tst_pred = gpmlp.gp_predict(tst_X)

            # print "CV:%d, GP #Selection:%d, GP MLP TR %s:%f, TST %s:%f" % (noise_d, len(gpmlp.active_d),
            #     metrics_style, metrics(tr_y, tr_pred, metrics_style),
            #     metrics_style, metrics(tst_y, tst_pred, metrics_style))
            #noise_mse[-1] += metrics(tst_y, tst_pred, metrics_style)
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
