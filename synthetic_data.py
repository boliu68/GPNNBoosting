import numpy as np
import mlp

from sknn.mlp import Regressor, Layer

def generate(N, hidden_len, noisy_d, true_dim):

    #Generate Synthetic Data using Feedforward Neural Networkw

    #assert true_dim <= d

    #Generate MLP with
    init_param = 1
    X = np.random.uniform(low=0, high=1, size=(N, true_dim))

    nn = Regressor(
                    layers = [
                        Layer("Rectifier", units=hidden_len),
                        Layer("Rectifier", units=hidden_len / 2),
                        Layer("Linear")
                    ],
                    learning_rate=0.00001,
                    n_iter=100,
                    regularize="L2",
                    weight_decay=0.005)
    nn.fit(X,np.random.rand(X.shape[0]))

    nn.set_parameters([(np.random.normal(loc=0, scale=init_param, size=(true_dim, hidden_len)), np.random.rand(hidden_len)),
        (np.random.normal(loc=0, scale=init_param, size=(hidden_len, hidden_len/2)), np.random.rand(hidden_len/2)),
        (np.random.normal(loc=0, scale=init_param, size=(hidden_len/2,1)), np.random.rand(1))])

    y = nn.predict(X)
    if noisy_d == 0:
        noisy_X = X
    else:
        noisy_X = np.hstack([X, np.random.uniform(low=0, high=1, size=(N, noisy_d))])

    return noisy_X, y.flatten(), nn

    #nn.fit(np.hstack([np.ones((tr_X.shape[0], 1)), tr_X[:, ls.coef_!=0]]), tr_y)#-ls.predict(tr_X))
    #tr_pred = nn.predict(np.hstack([np.ones((tr_X.shape[0], 1)), tr_X[:, ls.coef_!=0]])).flatten() #+ ls.predict(tr_X).flatten()
    #tst_pred = nn.predict(np.hstack([np.ones((tst_X.shape[0], 1)), tst_X[:, ls.coef_!=0]])).flatten() #+ ls.predict(tst_X).flatten()

if __name__ == '__main__':

    generate(N=100, hidden_len=10, noisy_d=20, true_dim=20)
