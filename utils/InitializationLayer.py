from tensorflow.keras.layers import Layer
import tensorflow as tf
from Muestreos import *
import numpy as np
import cv2
from loss_and_metrics import Between

class FSI_Initial(Layer):
    def __init__(self, p = 6, name="FSI_initial"):
        super(FSI_Initial, self).__init__(name=name)
        self.p = p
        

    def build(self, input_shape, k_size=5):
        super(FSI_Initial, self).build(input_shape[0])
        self.S = input_shape[0]
        self.M = tf.constant(self.S[1] * self.S[2] * self.S[3], dtype=tf.float32)
        self.R = tf.cast(tf.math.ceil(tf.math.divide(self.M, self.p)), dtype=tf.float32)

        Z0_abs = tf.random.normal(shape=(self.S[1], self.S[2]), mean=0.5, stddev=0.1)
        Z0_angle= tf.random.normal(shape=(self.S[1], self.S[2]), mean=0.0, stddev=(0.5*np.pi**(1/2)))
        Z0_abs = tf.cast(Z0_abs, tf.complex64)
        Z0_angle = tf.cast(Z0_angle, tf.complex64)
        Z0 = tf.multiply(Z0_abs, tf.math.exp(tf.multiply(Z0_angle, tf.constant(1j, dtype=tf.complex64))))
        self.Z0 = tf.math.divide(Z0, tf.norm(Z0, ord='fro',axis=(0,1)), name="Z0 initialization")

        

    def call(self, input):

        self.Masks = tf.cast(input[1], tf.complex64)
        Y = input[0]


        # INitializations
        Y = tf.cast(Y, dtype=tf.float32); Y = tf.transpose(Y,perm=[0,3,1,2])# (self.S[0], self.L, self.S[1], self.S[2])
        Z = tf.cast(tf.reshape(self.Z0, shape=(1, 1, self.S[1], self.S[2])), tf.complex64)
        div = tf.cast(tf.math.multiply(self.M,self.R), tf.complex64)



        # GET YTR
        S = tf.shape(Y)
        Y_S = tf.math.divide(Y,self.S[1])
        y_s = tf.reshape(Y_S, (S[0], S[1]*S[2]*S[3])) # Vectoriza
        y_s = tf.sort(y_s, axis = 1, direction='DESCENDING')
        aux  = tf.gather(y_s, indices=tf.cast(self.R-1, tf.int32), axis=1)
        threshold = tf.reshape(aux, (S[0], 1, 1, 1))
        Ytr = tf.reshape(tf.cast(tf.math.greater_equal(Y_S, threshold), dtype=Z.dtype) , S )
        return Ytr, self.Z0

class FSI_cell(Layer):
    def __init__(self, k_size = 5, name="Learnableinit", p = 6, init = False):
        super(FSI_cell, self).__init__(name=name)
        self.p = p
        self.k_size = k_size

        if init:
          # self.conv_real = tf.keras.layers.Conv2D(1, (self.k_size,self.k_size), padding="same", use_bias=False, trainable=True,name="FILTRO_ABS_INITIALIZATION", data_format='channels_first', kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.2, seed=None))
          # self.conv_imag = tf.keras.layers.Conv2D(1, (self.k_size,self.k_size), padding="same", use_bias=False, trainable=True,name="FILTRO_ANG_INITIALIZATION", data_format='channels_first', kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.2, seed=None))
          self.conv_real = tf.keras.layers.Conv2D(1, (self.k_size,self.k_size), padding="same", use_bias=False, trainable=True,name="FILTRO_ABS_INITIALIZATION", data_format='channels_first', kernel_constraint = Between(0,1))
          self.conv_imag = tf.keras.layers.Conv2D(1, (self.k_size,self.k_size), padding="same", use_bias=False, trainable=True,name="FILTRO_ANG_INITIALIZATION", data_format='channels_first', kernel_constraint = Between(0,1))
        else:
          self.conv_real = tf.keras.layers.Conv2D(1, (self.k_size,self.k_size), padding="same", use_bias=False, trainable=True,name="FILTRO_ABS_INITIALIZATION", data_format='channels_first', kernel_constraint = Between(0,1))
          self.conv_imag = tf.keras.layers.Conv2D(1, (self.k_size,self.k_size), padding="same", use_bias=False, trainable=True,name="FILTRO_ANG_INITIALIZATION", data_format='channels_first', kernel_constraint = Between(0,1))
        

    def build(self, input_shape, k_size=5):
        super(FSI_cell, self).build(input_shape[1])
        self.S = input_shape[0][1]


        self.A = lambda y: A_Fran(y, tf.reshape(self.Masks, (1,*self.Masks.shape)))
        self.AT = lambda y: AT_Fran(y,tf.reshape(self.Masks, (1,*self.Masks.shape)))       

    def call(self, input):


        self.Masks = tf.cast(input[0][0], tf.complex64)
        Ytr = input[0][1]
        Z = input[1]
        Z = tf.math.divide(self.AT(tf.multiply(Ytr, self.A(Z))),self.S[3]**2*self.S[1]**2*self.S[2]**2*self.p)

        Z_real_filt = self.conv_real(tf.math.real(Z)) - self.conv_imag(tf.math.imag(Z))
        Z_imag_filt = self.conv_imag(tf.math.real(Z)) + self.conv_real(tf.math.imag(Z))
        Z = tf.complex(Z_real_filt, Z_imag_filt)
        Z = tf.math.divide(Z, tf.norm(Z, ord='fro', axis=(2,3), keepdims=True))

        return Z
