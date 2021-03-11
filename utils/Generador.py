import tensorflow as tf
import numpy as np

class Encoder(tf.keras.layers.Layer):
    
    def __init__(self, n_filters=32, k_size=3, alpha = 0.5, name="Encoder", bloque = 1, **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.n_filters = n_filters
        self.k_size = k_size
        self.alpha = alpha
        self.conv_1 = tf.keras.layers.Conv2D(self.n_filters, self.k_size, padding="same", name= "DConv"+"_B"+str(bloque)+"_1")
        self.LRL1 = tf.keras.layers.LeakyReLU(alpha = self.alpha, name= "DLRL"+"_B"+str(bloque)+"_1")
        #self.drop_out = tf.keras.layers.Dropout(0.1, name= "Drop"+"_B"+str(bloque)+"_1")
        self.conv_2 = tf.keras.layers.Conv2D(self.n_filters, self.k_size, padding="same", name= "DConv"+"_B"+str(bloque)+"_2")
        self.LRL2 = tf.keras.layers.LeakyReLU(alpha = self.alpha, name= "DLRL"+"_B"+str(bloque)+"_2")
        self.conv_3 = tf.keras.layers.Conv2D(self.n_filters, self.k_size, padding="same", name= "DConv"+"_B"+str(bloque)+"_3")
        self.LRL3 = tf.keras.layers.LeakyReLU(alpha = self.alpha, name= "DLRL"+"_B"+str(bloque)+"_3")
        self.MP = tf.keras.layers.MaxPool2D((2, 2), (2, 2), name="MP_B_"+str(bloque))

    def call(self, inputs):
        c = self.conv_1(inputs)
        c = self.LRL1(c)
        #c = self.drop_out(c)
        c = self.conv_2(c)
        c = self.LRL2(c)
        c = self.conv_3(c)
        skip = self.LRL3(c)
        salida = self.MP(skip)
        return [skip, salida]

class Decoder(tf.keras.layers.Layer):
    
    def __init__(self, n_filters=32, k_size=3, alpha = 0.2, name="Decoder", bloque = 1,**kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.n_filters = n_filters
        self.k_size = k_size
        self.alpha = alpha
        self.conv_t = tf.keras.layers.Conv2DTranspose(self.n_filters, self.k_size, padding="same", strides = (2, 2), name= "UTConv"+"_B"+str(bloque)+"_1")
        self.LRL1 = tf.keras.layers.LeakyReLU(alpha = self.alpha, name= "ULRL"+"_B"+str(bloque)+"_1")
        self.conv_2 = tf.keras.layers.Conv2D(self.n_filters, self.k_size, padding="same", name= "UConv"+"_B"+str(bloque)+"_2")
        self.LRL2 = tf.keras.layers.LeakyReLU(alpha = self.alpha, name= "ULRL"+"_B"+str(bloque)+"_2")
        self.conv_3 = tf.keras.layers.Conv2D(self.n_filters, self.k_size, padding="same", name= "UConv"+"_B"+str(bloque)+"_3")
        self.LRL3 = tf.keras.layers.LeakyReLU(alpha = self.alpha, name= "ULRL"+"_B"+str(bloque)+"_3")
        self.concat = tf.keras.layers.Concatenate(name="Concat_B_"+str(bloque))

    def call(self, inputs):

        skip, entrada = inputs
        c = self.conv_t(entrada)
        up_sampled = self.LRL1(c)
        concat = self.concat([up_sampled, skip])

        c = self.conv_2(concat)
        c = self.LRL2(c)
        c = self.conv_3(c)
        salida = self.LRL3(c)
        return salida

# class Unet(tf.keras.layers.Layer):
    
#     def __init__(self, k_size=3, alpha = 0.5, name="UNET", **kwargs):
#         super(Unet, self).__init__(name=name, **kwargs)
        
#         self.k_size = k_size
#         self.alpha = alpha


#         # Augment channel 
#         self.aument_channel_conv =  tf.keras.layers.Conv2D(3,1,padding="same", name="aument_channel_conv")

#         # Encoder pretrained model
#         self.encoder = tf.keras.applications.VGG16(include_top=False, weights='imagenet')

#         for layer in self.encoder.layers[:-5]:
#           layer.trainable = False

#         # Index from the encoder skip layers
#         self.skip_layers_index  = np.array([ indx-1 for indx,layer in enumerate(self.encoder.layers) if "pool" in layer.name])

#         self.filtros = np.array([layer.filters for layer in np.array(self.encoder.layers)[self.skip_layers_index]])//2
#         # Bottleneck?
#         self.conv_1 = tf.keras.layers.Conv2D(self.filtros[-1], 1, padding="same", name = "Conv_BottleNEck_1")
#         self.LRL1 = tf.keras.layers.LeakyReLU(alpha=0.2, name = "B_LRL_1")
#         self.conv_2 = tf.keras.layers.Conv2D(self.filtros[-1], 1, padding="same", name = "Conv_BottleNEck_2")
#         self.LRL2 = tf.keras.layers.LeakyReLU(alpha=0.2, name = "B_LRL_2")


        

#         self.dencoder_1 = Decoder(self.filtros[4], name = "Decoder_1", bloque=2)
#         self.dencoder_2 = Decoder(self.filtros[3], name = "Decoder_2", bloque=1)
#         self.dencoder_3 = Decoder(self.filtros[2], name = "Decoder_3", bloque=1)
#         self.dencoder_4 = Decoder(self.filtros[1], name = "Decoder_4", bloque=1)
#         self.dencoder_5 = Decoder(self.filtros[0], name = "Decoder_5", bloque=1)
                          
#     def call(self, inputs):

#         aumented_chan = self.aument_channel_conv(inputs)

#         encoder_layers = self.encoder.layers

#         x = aumented_chan
#         skip_layers = []
#         for indx, layer in enumerate(encoder_layers):
#           x = layer(x)

#           if indx in self.skip_layers_index:
#             skip_layers.append(x)

#         salida_vgg = x

#         b = self.conv_1(salida_vgg)
#         b = self.LRL1(b)
#         b = self.conv_2(b)
#         b = self.LRL2(b)

#         c = self.dencoder_1([skip_layers[4], b])
#         c = self.dencoder_2([skip_layers[3], c])
#         c = self.dencoder_3([skip_layers[2], c])
#         c = self.dencoder_4([skip_layers[1], c])
#         salida_unet = self.dencoder_5([skip_layers[0], c])

        
#         return salida_unet

#     def model(self, input_shape = (256,256,2)):
#         x = tf.keras.Input(shape = input_shape)
#         return tf.keras.Model(inputs=[x], outputs=self.call(x))


class Unet(tf.keras.layers.Layer):
    
    def __init__(self, k_size=3, alpha = 0.5, name="UNET", **kwargs):
        super(Unet, self).__init__(name=name, **kwargs)
        
        self.k_size = k_size
        self.alpha = alpha

        self.filtros = [8, 16, 32, 64, 128]
        # Bottleneck?
        self.conv_1 = tf.keras.layers.Conv2D(self.filtros[-1], 1, padding="same", name = "Conv_BottleNEck_1")
        self.LRL1 = tf.keras.layers.LeakyReLU(alpha=0.2, name = "B_LRL_1")
        self.conv_2 = tf.keras.layers.Conv2D(self.filtros[-1], 1, padding="same", name = "Conv_BottleNEck_2")
        self.LRL2 = tf.keras.layers.LeakyReLU(alpha=0.2, name = "B_LRL_2")


        
        self.encoder_1 = Encoder(self.filtros[0], name = "Encoder_1", bloque=2)
        self.encoder_2 = Encoder(self.filtros[1], name = "Encoder_2", bloque=1)
        self.encoder_3 = Encoder(self.filtros[2], name = "Encoder_3", bloque=1)
        self.encoder_4 = Encoder(self.filtros[3], name = "Encoder_4", bloque=1)
        self.encoder_5 = Encoder(self.filtros[4], name = "Encoder_5", bloque=1)

        self.dencoder_1 = Decoder(self.filtros[4], name = "Decoder_1", bloque=2)
        self.dencoder_2 = Decoder(self.filtros[3], name = "Decoder_2", bloque=1)
        self.dencoder_3 = Decoder(self.filtros[2], name = "Decoder_3", bloque=1)
        self.dencoder_4 = Decoder(self.filtros[1], name = "Decoder_4", bloque=1)
        self.dencoder_5 = Decoder(self.filtros[0], name = "Decoder_5", bloque=1)
                          
    def call(self, inputs):


        s0,x = self.encoder_1(inputs)
        s1,x = self.encoder_2(x)
        s2,x = self.encoder_3(x)
        s3,x = self.encoder_4(x)
        s4,x = self.encoder_5(x)
 
        b = self.conv_1(x)
        b = self.LRL1(b)
        b = self.conv_2(b)
        b = self.LRL2(b)

        c = self.dencoder_1([s4, b])
        c = self.dencoder_2([s3, c])
        c = self.dencoder_3([s2, c])
        c = self.dencoder_4([s1, c])
        salida_unet = self.dencoder_5([s0, c])

        
        return salida_unet

    def model(self, input_shape = (256,256,2)):
        x = tf.keras.Input(shape = input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

