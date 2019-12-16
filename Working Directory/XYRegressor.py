import tensorflow as tf


class XYRegressor(tf.keras.Model):
    def __init__(self):
        super(XYRegressor, self).__init__()
        self.dx1 = tf.keras.layers.Dense(64, activation='relu', input_shape=(17,))
        self.dx2 = tf.keras.layers.Dense(64, activation='relu', 
                                         kernel_regularizer=tf.keras.regularizers.l2(0.005), 
                                         bias_regularizer=tf.keras.regularizers.l2(0.005))      
        self.dx3 = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.005), 
                                         bias_regularizer=tf.keras.regularizers.l2(0.005))
        
        self.dy1 = tf.keras.layers.Dense(64, activation='relu', input_shape=(17,))
        self.dy2 = tf.keras.layers.Dense(64, activation='relu', 
                                         kernel_regularizer=tf.keras.regularizers.l2(0.005), 
                                         bias_regularizer=tf.keras.regularizers.l2(0.005))
        self.dy3 = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.005), 
                                         bias_regularizer=tf.keras.regularizers.l2(0.005))
        
        self.concat = tf.keras.layers.Concatenate(axis=1)

    def call(self, i):
        x = self.dx1(i)
        x = self.dx2(x)
        x = self.dx3(x)
        
        y = self.dy1(i)
        y = self.dy2(y)
        y = self.dy3(y)
        return self.concat([x, y])