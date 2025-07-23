# coding=utf-8
# vim: sw=4 et tw=100
"""
callback

written by Yuwei Fan (ywfan@stanford.edu)
"""

from __future__ import print_function
import timeit
import numpy as np
from keras.callbacks import Callback
from keras import backend as K

class SaveBestModel(Callback):
    """Save the best model
    # Arguments
        filename: string to save the model file.
        verbose: verbosity mode, 0 or 1.
        period: Interval (number of epochs) between checkpoints.
    """
    def __init__(self, filename, check_result=None, verbose=1, period=1,
                 output=print, patience=10000, model_for_save=None, test_weight=0.5,
                 reduceLR=False, min_lr=None, patience_lr=None, factor=0.5):
        super(SaveBestModel, self).__init__()
        self.filename               = filename
        self.check_result           = check_result
        self.verbose                = verbose
        self.period                 = period
        self.output                 = output
        self.best_epoch             = 0
        self.best_epoch_update      = 0
        self.epochs_since_last_save = 0
        self.best_err_train         = 1
        self.best_err_test          = 1
        self.best_err_train_max     = 1
        self.best_err_test_max      = 1
        self.best_err_var_train     = 1
        self.best_err_var_test      = 1
        self.start                  = timeit.default_timer()
        self.patience               = patience
        self.stop_epoch             = 0
        self.model_for_save         = model_for_save
        self.test_weight            = max(min(test_weight, 1), 0)
        self.reduceLR               = reduceLR
        self.min_lr                 = min_lr
        self.patience_lr            = patience_lr or patience // 2
        self.factor                 = factor
        self.err_history            = []

    def on_train_begin(self, logs=None):
        self.compare_with_best_model(-1)
        if self.reduceLR:
            self.patience_lr = min(self.patience_lr, self.patience)
            if self.min_lr is None:
                self.min_lr = self.model.optimizer.lr / 10
                self.min_lr = max(self.min_lr, 1e-5)
            self.output("lr = %f" % self.model.optimizer.lr)
            self.output("min_lr = %f" % self.min_lr)
            self.output("patience = %d" % self.patience)
            self.output("patience lr = %d" % self.patience_lr)

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            self.compare_with_best_model(epoch)

        if self.reduceLR and (epoch + 1 - self.best_epoch_update) >= self.patience_lr:
            old_lr = self.model.optimizer.lr
            if old_lr > self.min_lr + 1e-7:
                new_lr = old_lr * self.factor
                new_lr = max(new_lr, self.min_lr)
                self.model.optimizer.lr = new_lr
                self.output('\nEpoch %05d: Reducing learning rate from %fto %s.' % (epoch + 1,
                                                                                    old_lr, new_lr))
                if self.model_for_save is None:
                    self.model.load_weights(self.filename, by_name=False)
                else:
                    self.model_for_save.logs(self.filename, by_name=False)
                self.best_epoch = epoch + 1
                self.best_epoch_update = epoch + 1

        if (epoch + 1 - self.best_epoch_update) >= self.patience:
            self.model.stop_training = True
            self.stop_epoch = epoch
            self.output('Early stoped at epoch %d with patience %d'
                        % (self.stop_epoch, self.patience))

    def compare_with_best_model(self, epoch):
        t1 = timeit.default_timer()
        err_train, err_test = self.check_result(self.model)
        self.err_history.append([epoch+1, np.mean(err_train), np.mean(err_test)])
        tw = self.test_weight
        err_old = self.best_err_train * (1 - tw) + self.best_err_test * tw
        err_new = np.mean(err_train) * (1 - tw) + np.mean(err_test) * tw
        if err_old > err_new and abs(err_new / err_old - 1) > 1.e-3:
            self.best_epoch         = epoch + 1
            self.best_epoch_update  = epoch + 1
            self.best_err_train     = np.mean(err_train)
            self.best_err_test      = np.mean(err_test)
            self.best_err_train_max = np.amax(err_train)
            self.best_err_test_max  = np.amax(err_test)
            self.best_err_var_train = np.var(err_train)
            self.best_err_var_test  = np.var(err_test)
            # save weights of the model
            if self.model_for_save is None:
                self.model.save_weights(self.filename, overwrite=True)
            else:
                self.model_for_save.save_weights(self.filename, overwrite=True)

        if self.verbose == 1:
            t2 = timeit.default_timer()
            self.output("Epoch %d:\t runtime of prediction = %.2f secs" % ((epoch + 1), (t2 - t1)))
            self.output("ave/max error of train/test data:\t %.2e %.2e \t %.1e %.1e " %
                        (np.mean(err_train), np.mean(err_test),
                         np.amax(err_train), np.amax(err_test)))
            self.output('best train/test error = %.2e, %.2e\t at epoch = %d,\t fit time = %.1f secs'
                        % (self.best_err_train, self.best_err_test,
                           self.best_epoch, (t2 - self.start)))
            self.output('best train/test error: var / max = %.1e, %.1e, %.1e, %.1e'
                        % (self.best_err_var_train, self.best_err_train_max,
                           self.best_err_var_test, self.best_err_test_max))
        elif self.verbose == 2:
            t2 = timeit.default_timer()
            self.output("error: (%.4g, %.4g), best error: (%.4g, %.4g) at epoch %d, fit time = %.1f secs" %
                        (np.mean(err_train), np.mean(err_test),
                         self.best_err_train, self.best_err_test,
                         self.best_epoch, (t2 - self.start)))
