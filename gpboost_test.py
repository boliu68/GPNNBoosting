import math
import sys
import pickle
import numpy as np
from scipy import io
from utils import metrics
from mx_GPBoost import mx_gp_mlp
from multiprocessing import Pool
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import Lasso, Ridge, LassoCV, LogisticRegression as LogR, LogisticRegressionCV as LogRCV

def parallel_gpboost(p):

    #dict(tr=tr.copy(), tst=tst.copy(), hdl=hdl, reg=rg, depth=-1, cv_id=cv_id, mode="hdl")
    tr = p["tr"]
    tst = p["tst"]

    tr_X = gX[tr, :]
    tr_y = gy[tr]

    tst_X = gX[tst, :]
    tst_y = gy[tst]

    hdl = p["hdl"]
    reg = p["reg"]
    depth = p["depth"]
    cv_id = p["cv_id"]
    mode = p["mode"]

    cpu_id = p["cpu_id"]
    print "Training CV ID:%d, CPU ID:%d, Mode:%s" % (cv_id, cpu_id, mode)
    out.write("Training CV ID:%d, CPU ID:%d, Mode:%s" % (cv_id, cpu_id, mode))
    out.flush()
    ###############################################
    ###########Fit MX GP Boosting #################

    #print "Training Cv:%d" % cv_id
    if mode == "hdl":
        mx_mlp = mx_gp_mlp(hidden_len=[hdl, 2], reg=reg, momentum=0.9, init_param=1, activation_func="RELU", metrics_func=metrics_style)
        mx_mlp.gp_fit(tr_X,tr_y, gp_lambda=0, max_features=30, lr=0.05, max_iter=1000, debug=False, tst_X=tst_X, tst_y=tst_y, cpu_id=cpu_id)
    elif mode == "dpt_i":
        dpt_list = [10 * (i+1) for i in range(depth)]
        dpt_list += [2]
        mx_mlp = mx_gp_mlp(hidden_len=dpt_list, reg=reg, momentum=0.9, init_param=1, activation_func="RELU", metrics_func=metrics_style)
        mx_mlp.gp_fit(tr_X,tr_y, gp_lambda=0, max_features=30, lr=0.05, max_iter=1000, debug=False, tst_X=tst_X, tst_y=tst_y, cpu_id=cpu_id)
    elif mode == "dpt_d":
        dpt_list = [30 / (i+1) for i in range(depth)]
        dpt_list += [2]
        mx_mlp = mx_gp_mlp(hidden_len=dpt_list, reg=reg, momentum=0.9, init_param=1, activation_func="RELU", metrics_func=metrics_style)
        mx_mlp.gp_fit(tr_X,tr_y, gp_lambda=0, max_features=30, lr=0.05, max_iter=1000, debug=False, tst_X=tst_X, tst_y=tst_y, cpu_id=cpu_id)

    return dict(param=p,
        cv_id=cv_id,
        max_grad=mx_mlp.max_grad_list,
        tr_loss=mx_mlp.tr_loss_list,
        tst_loss=mx_mlp.tst_loss_list,
        tr_metr=mx_mlp.tr_metr_list,
        tst_metr=mx_mlp.tst_metr_list)

def load_data():

    data_dict = io.loadmat("SMK-CAN-187.mat")
    X = data_dict["X"]
    y = data_dict["Y"].flatten()
    X.dtype=np.float64
    y[y==2] = 0

    X = X - np.tile(np.mean(X, 0), (X.shape[0], 1))

    return X, y

if __name__ == "__main__":

    global gX
    global gy

    gX, gy = load_data()
    metrics_style = "auc"

    cpu_id = int(sys.argv[1])
    print "CPU ID:%d" % cpu_id

    ########Begin 5 Fold Cross Validation

    lasso_result_list = []

    hidden_lens = [5, 10, 20, 50, 100]
    depth = [1,2,3,4]
    reg = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 0.1]

    gpboost_param_list = []

    kf = KFold(gX.shape[0], n_folds=5, shuffle=True)

    tr, tst = pickle.load(open("tr_tst.pkl"))[cpu_id]
    cv_id= cpu_id+ 1
    #for tr, tst in kf:
    if True:

        #Split Training and Test
        tr_X = gX[tr, :]
        tr_y = gy[tr]

        tst_X = gX[tst,:]
        tst_y = gy[tst]

        ###############################################
        #Training Lasso
        #ls = LogRCV(penalty="l1", Cs=50, fit_intercept=True, cv=5, n_jobs=-1, refit=True, solver="liblinear", scoring="log_loss")
        #ls.fit(tr_X, tr_y)
        #tr_pred = ls.predict_proba(tr_X)[:, 1]
        #tst_pred = ls.predict_proba(tst_X)[:, 1]

        #print "Lasso CPU ID:%d, CV:%d, Lasso, NNZ:%d, Tr Log Loss:%f, Tst Log Loss:%f, Lasso TR %s:%f, TST %s:%f" % (
        #    cpu_id, cv_id,
        #    np.sum(ls.coef_!=0),
        #    metrics(tr_y, tr_pred, "logistic"),
        #    metrics(tst_y, tst_pred, "logistic"),
        #    metrics_style, metrics(tr_y, tr_pred, metrics_style),
        #    metrics_style, metrics(tst_y, tst_pred, metrics_style))
        #lasso_result = dict(cv_id=cv_id,
        #    nnz=np.sum(ls.coef_!=0),
        #    tr_log=metrics(tr_y, tr_pred, "logistic"),
        #    tst_log=metrics(tst_y, tst_pred, "logistic"),
        #    tr_metr=metrics(tr_y, tr_pred, metrics_style),
        #    tst_metr=metrics(tst_y, tst_pred, metrics_style))

        #lasso_result_list.append(lasso_result)

	global out 
	out = open("Log_%d" % cpu_id, "w")

        ######Save GP Boost Parameter
        for hdl in hidden_lens:
            for rg in reg:
                gpboost_param_list += [dict(tr=tr.copy(), tst=tst.copy(), hdl=hdl, reg=rg, depth=-1, cv_id=cv_id, mode="hdl", cpu_id=cpu_id)]

        for dpt in depth:
            for rg in reg:
                gpboost_param_list += [dict(tr=tr.copy(), tst=tst.copy(), hdl=-1, reg=rg, depth=dpt, cv_id=cv_id, mode="dpt_i", cpu_id=cpu_id)]

        for dpt in depth:
            for rg in reg:
                gpboost_param_list += [dict(tr=tr.copy(), tst=tst.copy(), hdl=-1, reg=rg, depth=dpt, cv_id=cv_id, mode="dpt_d", cpu_id=cpu_id)]

    res = []
    #out.write("Total Combination:%d\n" % len(gpboost_param_list))
    for p in gpboost_param_list:
        out.write("Process %d / %d\n" % (len(res), len(gpboost_param_list)))
	out.flush()
	res.append(parallel_gpboost(p))
	pickle.dump(res, open("cpu%d.pkl" % cpu_id, "w"))
