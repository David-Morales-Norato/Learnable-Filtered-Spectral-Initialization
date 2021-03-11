import tensorflow as tf 
from Muestreos import A_Fran, AT_Fran
from AcquisitionLayer import Muestreo
from InitializationLayer import FSI_Initial,FSI_cell
from pruebas_layers import customGaussian
from InitializationUnrolled import InitializationUnrolled
from Generador import Unet
import numpy as np



# # 

# class ProposedInitializationModelUnrolled(tf.keras.Model):
#     def __init__(self, snapshots, SNR,p = 6, measures_type = "FRAN", use_generator = True):
#         #tf.print("init ProposedInitializationModelUnrolled")
#         super(ProposedInitializationModelUnrolled, self).__init__()
#         self.p = p
#         self.L = snapshots
#         self.measures_type = measures_type
#         self.Muestreo = Muestreo(self.L, name = "MuestreoLayer")
#         self.snr = SNR
#         if SNR != None:
#           self.gaus_noise = customGaussian(SNR, name = "gaussian_noise")
#         self.init_initialzation = FSI_Initial(p = self.p, name = "init_initialzation")
#         self.Initialation = FSI_cell(p = self.p, name = "InitializationLayer")
        
#     def build(self, input_shape):
#         #tf.print("build ProposedInitializationModelUnrolled")  
#         self.S = input_shape
#         self.M = tf.constant(self.S[1] * self.S[2] * self.L, dtype=tf.float32)
#         super(ProposedInitializationModelUnrolled, self).build(input_shape)
        
#         #tf.print(self.S)

#     def call(self, input):

#         [muestras, codigo] = self.Muestreo(input)
#         if self.snr != None:
#           #tf.print("Filtró")
#           muestras = self.gaus_noise(muestras)

#         normest = tf.math.sqrt(tf.math.divide(tf.reduce_sum(muestras, axis=(1,2,3), keepdims=True), tf.cast(self.S[3]*self.S[1]*self.S[2], tf.float32)))
#         Ytr, Z0 = self.init_initialzation([muestras, codigo])
#         Z1 = self.Initialation([[codigo, Ytr], Z0])
#         Z2 = self.Initialation([[codigo, Ytr], Z1])
#         Z3 = self.Initialation([[codigo, Ytr], Z2])
#         Z4 = self.Initialation([[codigo, Ytr], Z3])
#         Z5 = self.Initialation([[codigo, Ytr], Z4])
#         Z6 = self.Initialation([[codigo, Ytr], Z5])
#         Z7 = self.Initialation([[codigo, Ytr], Z6])
#         Z8 = self.Initialation([[codigo, Ytr], Z7])
#         Z9 = self.Initialation([[codigo, Ytr], Z8])
#         Z = self.Initialation([[codigo, Ytr], Z9])


#         Z = tf.multiply(tf.transpose(Z, perm=[0, 2,3,1]), tf.cast(normest, tf.complex64))

#         salida = [tf.squeeze(tf.math.abs(Z) ),  tf.squeeze(tf.math.angle(Z))]

#         return salida

#     def model(self):
#         x = tf.keras.Input(shape = self.S[1:])
#         return tf.keras.Model(inputs=[x], outputs=self.call(x))
