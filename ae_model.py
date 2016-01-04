# pylint: skip-file
import math
import mxnet as mx
import numpy as np
import logging
import data
from autoencoder import AutoEncoderModel, model
from sklearn.linear_model import Lasso, Ridge, LassoCV, LogisticRegressionCV as LogRCV

class ae_lr:

    def __init__(self, units, activation="Sigmoid", type="denoising"):

        logging.basicConfig(level=logging.DEBUG)
        self.ae_model = AutoEncoderModel(
            mx.gpu(0),
            [units,
            int(math.ceil(units/5.0)),
            int(math.ceil(units/20.0)),
            int(math.ceil(units/80.0))
            ],
            pt_dropout=0.2,
            internal_act='relu',
            output_act='relu')
        self.ls = LogRCV(penalty="l1", Cs=50, fit_intercept=True, cv=5, n_jobs=-1, refit=True, solver="liblinear", scoring="log_loss")

    def fit(self, X, y):

        #Layerwise PreTrain
        print "Layerwise Pretrain ..."
        self.ae_model.layerwise_pretrain(train_X, 256, 10000, 'sgd', l_rate=0.1, decay=0.01,
                             lr_scheduler=mx.misc.FactorScheduler(20000,0.1))
        print "Fine Tuning ..."
        self.ae_model.finetune(train_X, 256, 10000, 'sgd', l_rate=0.1, decay=0.01,
                   lr_scheduler=mx.misc.FactorScheduler(20000,0.1))

        low_X = self.apply(X)
        self.ls.fit(low_X, y)

    def predict(self, X, y):

        low_X = self.apply(X)
        return self.ls.predict(low_X)

    def predict_proba(self,X,y):

        low_X = self.apply(X)
        return self.ls.predict_proba(low_X)

    def apply(self, X):
        batch_size = 100
        data_iter = mx.io.NDArrayIter({'data': X}, batch_size=batch_size, shuffle=False,
                                      last_batch_handle='pad')
        Y = model.extract_feature(self.ae_model.loss, self.ae_model.args, self.ae_model.auxs, data_iter,
                                 X.shape[0], self.xpu).values()[0]
        return Y


