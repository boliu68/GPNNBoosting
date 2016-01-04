import math
from sknn import ae, mlp
import numpy as np
from sklearn.linear_model import Lasso, Ridge, LassoCV, LogisticRegressionCV as LogRCV
from dnn import SdA


class ae_lr:

    def __init__(self, units, activation="Sigmoid", type="denoising"):

        print units
        self.dae = ae.AutoEncoder(
                layers=[
                ae.Layer(activation, units=units)],
                #ae.Layer(activation, units=int(math.ceil(units/2.0))),
                #ae.Layer(activation, units=int(math.ceil(units/4.0)))],
                learning_rate=0.002,
                n_iter=10)

        self.ls = LogRCV(penalty="l1", Cs=50, fit_intercept=True, cv=5, n_jobs=-1, refit=True, solver="liblinear", scoring="log_loss")

    def fit(self, X, y):

        self.dae.fit(X)
        low_X = self.dae.transform(X)

        self.ls.fit(low_X, y)

    def predict(self, X, y):
        low_X = self.dae.transform(X)
        return self.ls.predict(low_X)

    def predict_proba(self,X,y):

        low_X = self.dae.transform(X)
        return self.ls.predict_proba(low_X)
