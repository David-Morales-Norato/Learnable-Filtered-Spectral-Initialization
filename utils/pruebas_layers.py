from tensorflow.keras.layers import Layer
import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from Muestreos import A_Fran, AT_Fran
from AcquisitionLayer import Muestreo 
import tensorflow as tf 
from Muestreos import A_Fran, AT_Fran
from AcquisitionLayer import Muestreo
from InitializationLayer import FSI_Initial,FSI_cell
from InitializationUnrolled import InitializationUnrolled
from Generador import Unet
import numpy as np


class customGaussian(Layer):
    def __init__(self, snr, name="Gaussian_noise_layer"):
        super(customGaussian, self).__init__(name=name)
        self.snr = snr
        
    def add_noise_each_x(self, x):

        m = x.shape[0]*x.shape[1]*x.shape[2]
        #tf.print("M", x.shape)
        divisor = m*10**(self.snr/10)
        #tf.print("DIVISOR: ", divisor)
        stddev = tf.math.sqrt(tf.math.divide(tf.math.pow(tf.norm(x, 'fro', axis= [0,1]),2), divisor))
        #tf.print("STDEV", self.snr)
        #tf.print("STDEV", stddev)
        return x + tf.keras.backend.random_normal(shape=x.shape,mean=0,stddev=stddev)

    def call(self, input):
        salida = tf.map_fn(self.add_noise_each_x,input)
        #tf.print(salida.shape)
        return salida

# # 

class ProposedInitializationModelUnrolledWithUnet(tf.keras.Model):
    def __init__(self, snapshots, SNR,p = 6, measures_type = "FRAN", use_generator = True, init = False):
        #tf.print("init ProposedInitializationModelUnrolled")
        super(ProposedInitializationModelUnrolledWithUnet, self).__init__()
        self.p = p
        self.L = snapshots
        self.measures_type = measures_type
        self.Muestreo = Muestreo(self.L, name = "MuestreoLayer")
        self.snr = SNR
        if SNR != None:
          self.gaus_noise = customGaussian(SNR, name = "gaussian_noise")
        self.init_initialzation = FSI_Initial(p = self.p, name = "init_initialzation")
        self.Initialation = FSI_cell(p = self.p, name = "InitializationLayer", init = init)
        self.use_generator = use_generator
        
        
        if (use_generator):
          self.pi = tf.constant(np.pi)
          self.Unet_abs = Unet(name = "Unet_ABS")
          self.Unet_ang = Unet(name = "Unet_ANG")
          self.nomralize = tf.keras.layers.experimental.preprocessing.Rescaling(1/(2*self.pi), offset=0.5)#tf.keras.layers.Lambda(lambda x: (x+self.pi)/(2*self.pi), name='Normalize')
          #self.unormalize = tf.keras.layers.Lambda(lambda x: (x*2*self.pi)-self.pi, name='Unormalize')

          # self.conv_1 = tf.keras.layers.Conv2D(8, 1, padding="same")
          # self.LRL1 = tf.keras.layers.LeakyReLU(alpha = 0.2)

          self.conv_abs_1 = tf.keras.layers.Conv2D(3, 1, padding="same", name = "ABS_improver1")
          self.LRL1_abs = tf.keras.layers.LeakyReLU(alpha = 0.2)
          self.conv_2_abs = tf.keras.layers.Conv2D(1, 1, padding="same", name = "ABS_improver2")
          self.LRL2_abs = tf.keras.layers.LeakyReLU(alpha = 0.2)
          self.conv_1_ang = tf.keras.layers.Conv2D(3, 1, padding="same", name = "ANG_improver1")
          self.LRL2_ang = tf.keras.layers.LeakyReLU(alpha = 0.2)
          self.conv_2_ang = tf.keras.layers.Conv2D(1, 1, padding="same", name = "ANG_improver2")
          self.LRL2_ang = tf.keras.layers.LeakyReLU(alpha = 0.2)

    def build(self, input_shape):
        #tf.print("build ProposedInitializationModelUnrolled")  
        self.S = input_shape
        self.M = tf.constant(self.S[1] * self.S[2] * self.L, dtype=tf.float32)
        super(ProposedInitializationModelUnrolledWithUnet, self).build(input_shape)
        
        #tf.print(self.S)

    def call(self, input):

        [muestras, codigo] = self.Muestreo(input)
        if self.snr != None:
          #tf.print("Filtró")
          muestras = self.gaus_noise(muestras)

        normest = tf.math.sqrt(tf.math.divide(tf.reduce_sum(muestras, axis=(1,2,3), keepdims=True), tf.cast(self.S[3]*self.S[1]*self.S[2], tf.float32)))
        Ytr, Z0 = self.init_initialzation([muestras, codigo])
        Z1 = self.Initialation([[codigo, Ytr], Z0])
        Z2 = self.Initialation([[codigo, Ytr], Z1])
        Z3 = self.Initialation([[codigo, Ytr], Z2])
        Z4 = self.Initialation([[codigo, Ytr], Z3])
        Z5 = self.Initialation([[codigo, Ytr], Z4])
        Z6 = self.Initialation([[codigo, Ytr], Z5])
        Z7 = self.Initialation([[codigo, Ytr], Z6])
        Z8 = self.Initialation([[codigo, Ytr], Z7])
        Z9 = self.Initialation([[codigo, Ytr], Z8])
        Z = self.Initialation([[codigo, Ytr], Z9])


        Z6 = tf.multiply(tf.transpose(Z6, perm=[0, 2,3,1]), tf.cast(normest, tf.complex64))
        Z7 = tf.multiply(tf.transpose(Z7, perm=[0, 2,3,1]), tf.cast(normest, tf.complex64))
        Z8 = tf.multiply(tf.transpose(Z8, perm=[0, 2,3,1]), tf.cast(normest, tf.complex64))
        Z9 = tf.multiply(tf.transpose(Z9, perm=[0, 2,3,1]), tf.cast(normest, tf.complex64))
        Z = tf.multiply(tf.transpose(Z, perm=[0, 2,3,1]), tf.cast(normest, tf.complex64))

        init_amp, init_ang = tf.squeeze(tf.math.abs(Z)),  tf.squeeze(tf.math.angle(Z))


        if (self.use_generator):

          
          phase = self.nomralize(init_ang)


          #init2 = tf.concat([tf.expand_dims(init_amp,-1), tf.expand_dims(phase,-1)], -1)
          amplitude = tf.expand_dims(init_amp,-1)
          phase = tf.expand_dims(phase,-1)
          salida_gene_abs  = self.Unet_abs(amplitude)
          salida_gene_ang  = self.Unet_ang(phase)

          # pred = self.conv_1(salida_gene)
          # final = self.LRL1(pred)


          amplitud = self.conv_abs_1(salida_gene_abs)
          amplitud = self.LRL1_abs(amplitud)
          amplitud = self.conv_2_abs(amplitud)
          amplitud = tf.squeeze(self.LRL2_abs(amplitud))




          angulo = self.conv_1_ang(salida_gene_ang)
          angulo = self.LRL2_ang(angulo)
          angulo = self.conv_2_ang(angulo)
          angulo = tf.squeeze(self.LRL2_ang(angulo))

          # #angulo = self.unormalize(angulo)
          # if tf.executing_eagerly():
          #   tf.print("")
          #   tf.print("Range init_ang", init_ang.numpy().min(),init_ang.numpy().max())
          #   tf.print("Range Phase", phase.numpy().min(),phase.numpy().max())
          #   tf.print("Range angulo", angulo.numpy().min(),angulo.numpy().max())

          # tf.print("hsape init_amp", init_amp.shape)
          # tf.print("hsape init_ang", init_ang.shape)
          # tf.print("hsape salida_gene_abs", salida_gene_abs.shape)
          # tf.print("hsape salida_gene_ang", salida_gene_ang.shape)

          salida = [init_amp , init_ang, amplitud , angulo]
          
          #salida = [init_amp , init_ang, tf.squeeze(tf.math.abs(Z9)) , tf.squeeze(tf.math.angle(Z9)), tf.squeeze(tf.math.abs(Z8)) , tf.squeeze(tf.math.angle(Z8)), tf.squeeze(tf.math.abs(Z7)) , tf.squeeze(tf.math.angle(Z7)), amplitud , angulo]
        else:
          
          #Z5 = tf.multiply(tf.transpose(Z5, perm=[0, 2,3,1]), tf.cast(normest, tf.complex64))
          #amplitud , angulo = tf.squeeze(tf.math.abs(26)) , tf.squeeze(tf.math.angle(26)) 
          
          salida = [init_amp , init_ang, tf.squeeze(tf.math.abs(Z9)) , tf.squeeze(tf.math.angle(Z9)), tf.squeeze(tf.math.abs(Z8)) , tf.squeeze(tf.math.angle(Z8)), tf.squeeze(tf.math.abs(Z7)) , tf.squeeze(tf.math.angle(Z7))]
        
        return salida

    def model(self):
        x = tf.keras.Input(shape = self.S[1:])
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

