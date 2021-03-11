from tensorflow.keras.layers import Layer
import tensorflow as tf
from Muestreos import *
import numpy as np


class GetYtr(Layer):
    def __init__(self, p):     
        self.p = p
        super(GetYtr, self).__init__()
   
    def build(self, input_shape):

        self.S = input_shape
        #print("S YTR SHAPE", self.S)
        self.L = self.S[1]
        self.M = tf.constant(self.S[2] * self.S[3] * self.L, dtype=tf.float32)
        self.R = tf.cast(tf.math.ceil(tf.math.divide(self.M, self.p)), dtype=tf.float32)


        super(GetYtr, self).build(self.S)

    def call(self, input):
        #print("GET YTR INPUTS", inputs.shape)
        #print()

        # GET YTR
        Y = input
        S = tf.shape(Y)
        Y_S = tf.math.divide(Y,self.S[1])
        y_s = tf.reshape(Y_S, (S[0], S[1]*S[2]*S[3])) # Vectoriza
        y_s = tf.sort(y_s, axis = 1, direction='DESCENDING')
        aux  = tf.gather(y_s, indices=tf.cast(self.R-1, tf.int32), axis=1)
        threshold = tf.reshape(aux, (S[0], 1, 1, 1))
        ytr = tf.reshape(tf.cast(tf.math.greater_equal(Y_S, threshold), dtype=tf.complex64) , S )

        #print("YTR OUTPUT: ", ytr.shape)
        return ytr
        
class UpdateZ(Layer):
    def __init__(self, p, no_layer, k_size=5): 
        self.k_size = k_size
        self.no_layer = no_layer
        self.p = p
        self.conv_layer = self.conv_layer = tf.keras.layers.Conv2D(1, (self.k_size,self.k_size), padding="same", use_bias=True, activation=None, trainable=True,name="CONV_LAYER"+str(self.no_layer), data_format='channels_first')
        super(UpdateZ, self).__init__()


    
    def build(self, input_shape):
        self.S = input_shape[0]
        self.L = self.S[3]
        self.M = tf.constant(self.S[1] * self.S[2] * self.L, dtype=tf.float32)
        self.R = tf.cast(tf.math.ceil(tf.math.divide(self.M, self.p)), dtype=tf.float32)
        self.div = tf.cast(tf.math.multiply(self.M,self.R), tf.complex64)

        self.A = lambda y: A_Fran(y, tf.reshape(self.Masks, (1,*self.Masks.shape)))
        self.AT = lambda y: AT_Fran(y,tf.reshape(self.Masks, (1,*self.Masks.shape)))

        super(UpdateZ, self).build(self.S)

    def call(self, input):
        self.Masks = input[1]
        
        Z = input[0]
        Ytr = input[2]

        #print("UPDATE Z input", Z.shape)

        # # A(z)
        # A_z = tf.signal.fft2d(tf.multiply(tf.math.conj(self.Masks), Z))


        # # AT(Z)
        # A_z = tf.multiply(Ytr, A_z)
        # mult_mass_z = tf.multiply(self.Masks, tf.signal.ifft2d(A_z))
        # res = tf.reduce_sum(mult_mass_z, axis=1, keepdims=True)

        # AT_z = tf.multiply(res, tf.cast(self.S[1]*self.S[2], dtype=res.dtype))

        # Z = tf.math.divide(AT_z, self.div)


        Z = tf.math.divide(self.AT(tf.multiply(Ytr, self.A(Z))), self.div)
        # Z = real + imag*j
        Z = tf.complex(self.conv_layer(tf.math.real(Z)), self.conv_layer(tf.math.imag(Z)))

        # Normalize
        Z = tf.math.divide(Z, tf.norm(Z, ord='fro', axis=(2,3), keepdims=True))
        #print("UPDATE Z output", Z.shape)
        return Z


class InitializationUnrolled(Layer):
    def __init__(self, npower_iter ,p = 6, k_size=5, name="Encoder"):
        #print("InitializationUnrolled")   
        self.npower_iter = npower_iter
        self.p = p
        self.k_size = k_size
        
        self.get_ytr = GetYtr(self.p)
        self.list_updates_z = [UpdateZ(self.p, i, self.k_size) for i in range(self.npower_iter)]
        super(InitializationUnrolled, self).__init__(name=name)
        

    
    def build(self, input_shape):
        
        self.S = input_shape[0]
        
        self.L = self.S[3]
        self.M = tf.constant(self.S[1] * self.S[2] * self.L, dtype=tf.float32)
        self.R = tf.cast(tf.math.ceil(tf.math.divide(self.M, self.p)), dtype=tf.float32)
        self.div = tf.cast(tf.math.multiply(self.M,self.R), tf.complex64)

        Z0 = tf.random.normal(shape=(self.S[1], self.S[2]), dtype=tf.float32)
        self.Z0 = tf.math.divide(Z0, tf.norm(Z0, ord='fro',axis=(0,1)))

        super(InitializationUnrolled, self).build(self.S)

    def call(self, input):

        #print("initialization unrolled input", inputs[0].shape)
        self.Masks = tf.cast(input[1], tf.complex64)
        Y = tf.cast(input[0], dtype=tf.float32); Y = tf.transpose(Y,perm=[0,3,1,2])# (tf.shape(y)[0], self.L, self.S[1], self.S[2])

        self.Ytr = self.get_ytr(Y)

        normest = tf.math.sqrt(tf.math.divide(tf.reduce_sum(Y, axis=(1,2,3), keepdims=True), tf.cast(self.M, tf.float32)))
        Z = tf.cast(tf.reshape(self.Z0, shape=(1, 1, self.S[1], self.S[2])), tf.complex64)

        
        #Updates
        for update_z in self.list_updates_z:
          Z = update_z([Z, self.Masks, self.Ytr])


        Z = tf.multiply(tf.transpose(Z, perm=[0, 2,3,1]), tf.cast(normest, tf.complex64))
        ZR = tf.cast(tf.concat([tf.math.abs(Z), tf.math.angle(Z)], -1), dtype=tf.float32)
        #$print("initialization unrolled output", ZR.shape)
        return ZR





        

