import math
import pickle
import mxnet as mx
import numpy as np
from utils import metrics
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

class mx_gp_mlp:

    def __init__(self, hidden_len=3, activation_func="RELU", reg=0, momentum=0.5, init_param=1, norm_style=None, metrics_func = "MSE"):

        self.hidden_len = hidden_len
        self.activation_func = activation_func

        self.hidden_unit = np.zeros(hidden_len)
        self.init_param = init_param

        self.active_d = []
        self.inactive_d = []

        self.norm_style = norm_style
        self.activation_func = activation_func
        self.metrics_func = metrics_func

        self.reg = reg
        self.momentum=momentum

        self.net = None

    def gp_fit(self, X, y, gp_lambda=0.1, max_features=-1, max_iter=50, lr=0.01, debug=False, tst_X=None, tst_y=None, lasso_model=None, cpu_id=0):

        assert X.shape[0] == len(y)

        # self.ls_X = X[:, lasso_model.coef_[0]!=0]
        # self.ls_X = np.hstack([np.ones((self.ls_X.shape[0], 1)), self.ls_X])
        # self.tst_ls_X = tst_X[:, lasso_model.coef_[0]!=0]
        # self.tst_ls_X = np.hstack([np.ones((self.tst_ls_X.shape[0], 1)), self.tst_ls_X])

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        tst_X = np.hstack([np.ones((tst_X.shape[0], 1)), tst_X])
	self.cpu_id = cpu_id
        #self.ls_X = np.zeros(X.shape)
        #self.tst_ls_X = np.zeros(tst_X.shape)


        #####Make Neural Network
        if self.net == None:
            self.stack_net()

        ####Bind the neural network
        self.net_exec = self.net.simple_bind(ctx=mx.cpu(cpu_id), data=X.shape, grad_req="write")
        self.tst_exec = self.net.simple_bind(ctx=mx.cpu(cpu_id), data=tst_X.shape)

        ####Assign the names and arrays to correspondent variable
        net_name = self.net_exec.arg_dict
        arg_names = self.net.list_arguments()
        data_name = "data"
        label_name = "softmax_label"
        ls_name = "ls"
        param_names = list(set(arg_names) - set([data_name, label_name, ls_name]))

        self.param_idx = [i for i in range(len(arg_names)) if arg_names[i] in param_names]
        self.data_idx = [i for i in range(len(arg_names)) if arg_names[i] ==  data_name][0]
        self.label_idx = [i for i in range(len(arg_names)) if arg_names[i] ==  label_name][0]
        #self.ls_name = [i for i in range(len(arg_names)) if arg_names[i] ==  ls_name][0]
        self.input_idx = [i for i in range(len(arg_names)) if arg_names[i] == "fc1_weight"][0]

        self.param_arrays = [self.net_exec.arg_arrays[i] for i in self.param_idx]
        self.grad_arrays = [self.net_exec.grad_arrays[i] for i in self.param_idx]
        self.data_arrays = self.net_exec.arg_dict[data_name]
        self.label_arrays = self.net_exec.arg_dict[label_name]
        #self.ls_arrays = self.net_exec.arg_dict[ls_name]

        self.tst_data_arrays = self.tst_exec.arg_dict[data_name]
        #self.tst_ls_arrays = self.tst_exec.arg_dict[ls_name]
        self.tst_param = [self.tst_exec.arg_arrays[i] for i in self.param_idx]

        self.input_param = self.net_exec.arg_arrays[self.input_idx]
        self.input_grad = self.net_exec.grad_arrays[self.input_idx]

        #Active Dim Currently
        self.active_d = range(1)
        self.inactive_d = range(1, X.shape[1])
        self.grad_rescaling = 1.0 / X.shape[0]
        self.init_lr = lr
        self.gp_lambda = gp_lambda

        #####Initialization
        initializer = mx.initializer.Xavier(factor_type="out", magnitude=2.34)
        for i in range(len(self.param_arrays)):
            if self.param_idx[i] == self.input_idx:
                empty_nd = mx.nd.zeros((self.param_arrays[i].shape[0], 1), mx.cpu(cpu_id))
                initializer(arg_names[self.param_idx[i]], empty_nd)
                tmp_param0 = np.zeros(self.param_arrays[i].shape)
                tmp_param0[:, 0] = empty_nd.asnumpy().flatten()
                self.param_arrays[i][:] = tmp_param0
            else:
                initializer(arg_names[self.param_idx[i]], self.param_arrays[i])

        #while(True):
        self.max_grad_list = []
        self.tr_loss_list = []
        self.tst_loss_list = []
        self.tr_metr_list = []
        self.tst_metr_list = []

        while (max_features == -1) or (len(self.active_d) <= max_features):

            #X_tr, X_vd, y_tr, y_vd = train_test_split(X, y, test_size=0.75)

            self.data_arrays[:] = X
            self.label_arrays[:] = y

            #self.ls_X[:, self.active_d] = X[:, self.active_d]
            #self.ls_arrays[:] = self.ls_X

            #Begin to Train
            it_loss = []
            it_loss_l2 = []

            lr = self.init_lr

            for it in range(max_iter):
                self.net_exec.forward(is_train=True)

                #Loss Calculated
                one_loss = metrics(y, self.net_exec.outputs[0].asnumpy()[:,1].flatten(), "logistic")
                it_loss.append(one_loss)
                for i in range(len(self.param_arrays)):
                    one_loss += self.reg * (np.linalg.norm(self.param_arrays[i].asnumpy()) ** 2)
                it_loss_l2.append(one_loss)
                #print "Log Loss:%f" % it_loss_l2[-1]

                self.net_exec.backward()

                for i in range(len(self.param_arrays)):
                    if self.param_idx[i] == self.input_idx:

                        tmp_param = self.param_arrays[i].asnumpy()
                        self.grad_arrays[i].wait_to_read()
                        grad_tmp = self.grad_arrays[i].asnumpy()

                        #Check Gradient NaN
                        if np.sum(np.isnan(grad_tmp)) > 0:
                            print "Found NaN"
                            self.max_grad_list.append(-1)
                            self.tr_loss_list.append(-1)
                            self.tst_loss_list.append(-1)
                            self.tr_metr_list.append(-1)
                            self.tst_metr_list.append(-1)
                            return

                        tmp_param += - lr * ( grad_tmp * self.grad_rescaling + self.reg * tmp_param)
                        tmp_param[:, self.inactive_d] = 0
                        self.param_arrays[i][:] = tmp_param
                    else:
                        self.param_arrays[i][:] += - lr * (self.grad_arrays[i] * self.grad_rescaling + self.reg * self.param_arrays[i])

                # try:
                #     if (it_loss[-1] > it_loss[-2]):
                #         lr = lr * 0.95
                #         #self.input_lr = self.input_lr * 0.9
                # except:
                #     pass

                if len(it_loss_l2) > 10:
                    if ((it_loss_l2[-10] - it_loss_l2[-1]) / it_loss_l2[-10] ) < 1e-6:
                        break
            if False:
                plt.close()
                plt.figure()
                plt.title("Current Active Features:%d" % (len(self.active_d)))
                plt.plot(it_loss, "b-", linewidth=2)
                plt.plot(it_loss_l2, "r-", linewidth=2)
                plt.savefig("fig/%d.jpg" % len(self.active_d))

            ####Output the loss
            tr_pred = self.net_exec.outputs[0].asnumpy()[:,1].flatten()

            #Test or Validation
            self.tst_data_arrays[:] = tst_X

            #self.tst_ls_X[:, self.active_d] = tst_X[:, self.active_d]
            #self.tst_ls_arrays[:] = self.tst_ls_X

            for i in range(len(self.param_arrays)):
                self.param_arrays[i].copyto(self.tst_param[i])
            self.tst_exec.forward(is_train=False)
            tst_pred = self.tst_exec.outputs[0].asnumpy()[:,1].flatten()

            #print "Iter:%d, Tr %s:%f, Tst %s:%f, Tr %s:%f, Tst %s:%f," % (len(self.active_d),
            #     "Log Loss", metrics(y, tr_pred, "logistic"),
            #     "Log Loss", metrics(tst_y, tst_pred, "logistic"),
            #     self.metrics_func, metrics(y, tr_pred, self.metrics_func),
            #     self.metrics_func, metrics(tst_y, tst_pred, self.metrics_func))

            max_grad = self.gp_add_feature()

            #Logging the training information
            self.max_grad_list.append(max_grad)
            self.tr_loss_list.append(metrics(y, tr_pred, "logistic"))
            self.tst_loss_list.append(metrics(tst_y, tst_pred, "logistic"))
            self.tr_metr_list.append(metrics(y, tr_pred, self.metrics_func))
            self.tst_metr_list.append(metrics(tst_y, tst_pred, self.metrics_func))

    def stack_net(self):

        if isinstance(self.hidden_len, int):
            print "The Unit length must be a list"
            exit()

        #ls = mx.symbol.Variable("ls")

        net = mx.symbol.Variable("data")

        for i in range(len(self.hidden_len) - 1):
            net = mx.symbol.FullyConnected(data=net, name="fc%d" % (i+1), num_hidden=self.hidden_len[i])
            net = mx.symbol.Activation(data=net, name='%s%d' % (self.activation_func, i+1), act_type=self.activation_func.lower())

        #net = mx.symbol.Concat(*[ls, net])

        net = mx.symbol.FullyConnected(data=net, name="fc%d" % (len(self.hidden_len)), num_hidden=self.hidden_len[-1])
        net = mx.symbol.SoftmaxOutput(data=net, name="softmax")
        self.net = net

    def gp_add_feature(self):

        np_input_grad = self.input_grad.asnumpy()
        inactive_norm_grad = np.linalg.norm(np_input_grad, axis=0, ord=self.norm_style)[self.inactive_d]
        inactive_id = np.argmax(inactive_norm_grad)

        #print "Max Gradient:%f" %  (inactive_norm_grad[inactive_id] * self.grad_rescaling * self.init_lr)

        if inactive_norm_grad[inactive_id] * self.init_lr <= self.gp_lambda:
            #End Training
            #print "Normal Error"
            #exit()
            return -1

        #print "Mean Hidden Unit:%f, Mean Grad:%f" % (np.mean(np.abs(self.hd)), np.mean(np.abs(np_input_grad)))
        #Add Feature
        self.active_d.append(self.inactive_d[inactive_id])
        self.inactive_d.pop(inactive_id)
        #self.input_lr[:, self.inactive_d[inactive_id]] = self.init_lr

        tmp_param = self.input_param.asnumpy()
        empty_param = mx.nd.empty((tmp_param.shape[0], 1), mx.cpu(self.cpu_id))
        initializer = mx.initializer.Xavier(factor_type="out", magnitude=2.34)
        initializer("fc1_weight", empty_param)
        tmp_param[:, self.active_d[-1]] = empty_param.asnumpy().flatten()

        self.input_param[:] = tmp_param

        return inactive_norm_grad[inactive_id] * self.grad_rescaling * self.init_lr


    def gp_predict(self, X):

        "Forward Neural Network that predict the label"
        #Not Yet Implemented


    def apply(self, X):
        act = activation(np.dot(X[:, self.active_d], self.W), self.activation_func)

        return act - np.tile(np.mean(act, 0), (act.shape[0], 1))

