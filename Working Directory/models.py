import pandas as pd
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, size='normal'):
        self.history = None

        if size == 'normal':
            self.create_model()
        elif size == 'small':
            self.create_small_model()

    def create_small_model(self):
        self.optimizer = tf.keras.optimizers.RMSprop(0.001)
        self.loss = "mse"
        self.metrics = ["mae", "mse"]
        self.batch_size = 32
        self.epochs = 1000
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=50, monitor='val_loss')
        ]
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(17,)),
            tf.keras.layers.Dense(2, 
                kernel_regularizer=tf.keras.regularizers.l2(0.02), 
                bias_regularizer=tf.keras.regularizers.l2(0.02), 
                activity_regularizer=tf.keras.regularizers.l2(0.02))
        ])

        self.model.compile(
            optimizer=self.optimizer, 
            loss=self.loss, 
            metrics=self.metrics)
        self.model.summary()

    def create_model(self):
        self.optimizer = tf.keras.optimizers.RMSprop(0.001)
        self.loss = "mse"
        self.metrics = ["mae", "mse"]
        self.batch_size = 32
        self.epochs = 1000
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=50, monitor='val_loss')
        ]
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(17,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(2, 
                kernel_regularizer=tf.keras.regularizers.l2(0.02), 
                bias_regularizer=tf.keras.regularizers.l2(0.02), 
                activity_regularizer=tf.keras.regularizers.l2(0.02))
        ])

        self.model.compile(
            optimizer=self.optimizer, 
            loss=self.loss, 
            metrics=self.metrics)
        self.model.summary()

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = tf.keras.models.load_model(filename)

    def fit(self, X_train, y_train, X_validation, y_validation):
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size, epochs=self.epochs,
            callbacks=self.callbacks, validation_data=(X_validation, y_validation))
        return self.history

    def predict(self, X):
        return self.model.predict(X)
    
    def show_history(self):
        if self.history is None:
            raise Exception("The model has not been trained")

        plt.figure()
        plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
        plotter.plot({"": self.history}, metric = "mae")
        plt.ylim([0, 10])
        plt.ylabel('Mean Absolute Error')

        plt.figure()
        plotter.plot({"": self.history}, metric = "mse")
        plt.ylim([0, 20])
        plt.ylabel('Mean Squared Error')

        plt.show()

    def model_validation(self, X_validation, y_validation):
        preds = self.model.predict(X_validation)

        fig = plt.figure(figsize=(10, 20))
        lims = [0, 50]
        ax1 = fig.add_subplot(121, aspect='equal', title='X', xlim=lims, ylim=lims, xlabel='True Values', ylabel='Predictions')
        ax1.scatter(y_validation[:, 0], preds[:, 0])
        ax1.plot(lims, lims)

        error = preds - y_validation
        ax2 = fig.add_subplot(122, xlabel='Prediction Error', ylabel='Count')
        ax2.hist(error[:, 0])

        fig = plt.figure(figsize=(10, 20))
        lims = [0, 50]
        ax1 = fig.add_subplot(121, aspect='equal', title='Y', xlim=lims, ylim=lims, xlabel='True Values', ylabel='Predictions')

        ax1.scatter(y_validation[:, 1], preds[:, 1])
        ax1.plot(lims, lims)

        error = preds - y_validation
        ax2 = fig.add_subplot(122, xlabel='Prediction Error', ylabel='Count')
        ax2.hist(error[:, 1])

        plt.show()

    def model_testing(self, X_test, y_test):
        test_scores = self.model.evaluate(X_test, y_test, verbose=2)
        print("Test loss:", test_scores[0])
        print("Mean absolute error:", test_scores[1])
        print("Mean squared error:", test_scores[2])