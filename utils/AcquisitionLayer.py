import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from Muestreos import *



class Muestreo(Layer):
    def __init__(self, snapshots, name="Encoder"):
        self.L = snapshots
        super(Muestreo, self).__init__(name=name)

    def build(self, input_shape):
        self.S = input_shape
        self.Masks = tf.convert_to_tensor(np.random.choice([1, -1, 1j, -1j], size=[self.L, *self.S[1:-1]]), dtype=tf.complex64)
        self.A = lambda y: A_Fran(y, tf.reshape(self.Masks, (1,*self.Masks.shape)))
        super(Muestreo, self).build(input_shape)

    def call(self, input):
        input = tf.cast(input, dtype=tf.float32)
        


        amplitude = tf.gather(input, indices=0, axis=3)
        phase = tf.gather(input, indices=1, axis=3)

        Z = tf.multiply(tf.cast(amplitude, dtype=tf.complex64), tf.math.exp(tf.multiply(tf.cast(phase, dtype=tf.complex64), tf.constant(1j, dtype=tf.complex64))))
        Z = tf.expand_dims(Z,1) 

        Y = tf.square(tf.math.abs(self.A(Z)))
        #Y = tf.square(tf.math.abs(tf.signal.fft2d(tf.multiply(tf.math.conj(self.Masks), Z))))
        Y = tf.transpose(Y,perm=[0,2,3,1])  # (self.S[0], self.S[1], self.S[2], self.L)

        return [Y, self.Masks]

