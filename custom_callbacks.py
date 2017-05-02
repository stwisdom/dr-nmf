import keras
import cPickle

class LossHistory(keras.callbacks.Callback):
    def __init__(self, histfile):
        self.histfile = histfile
    
    def on_train_begin(self, logs={}):
        self.metrics_on_batch_end = {}
        self.metrics_on_epoch_end = {}

    def on_batch_end(self, batch, logs={}):
        for key in logs.keys():
            if not (key in self.metrics_on_batch_end.keys()):
                self.metrics_on_batch_end[key] = [logs.get(key)]
            else:
                self.metrics_on_batch_end[key].append(logs.get(key))

    def on_epoch_end(self, epoch, logs={}):
        for key in logs.keys():
            if not (key in self.metrics_on_epoch_end.keys()):
                self.metrics_on_epoch_end[key] = [logs.get(key)]
            else:
                self.metrics_on_epoch_end[key].append(logs.get(key))

        with open(self.histfile, 'wb') as f:
            cPickle.dump({'on_batch_end': self.metrics_on_batch_end, 'on_epoch_end': self.metrics_on_epoch_end}, f)

