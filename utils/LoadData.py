from scipy.io import loadmat
import tensorflow as tf
import numpy as np
import os
import h5py



class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_paths_images, batch_size=128, dim=(256,256), n_channels=2,shuffle=False):
        'Initialization'
        #assert os.path.exists(path_images), "EL DIRECTORIO " + path_images + " NO EXISTE"

        self.list_paths_images = list_paths_images
        self.resize = tf.keras.layers.experimental.preprocessing.Resizing(*dim)
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dim = dim
        self.on_epoch_end()

    def __len__(self):
      'Denotes the number of batches per epoch'
      return (len(self.list_paths_images)//self.batch_size)

    #function to read hdf5 file
    def read_mat(self, path):
      data = None
      if (os.path.exists(path)):
        data = loadmat(path)
        data = np.array(data['DOI'])


        data[...,0] = data[...,0]/255
        min_x = data[...,1].min()
        max_x = data[...,1].max()
        data[...,1] = 2*np.pi*((data[...,1]-min_x)/(max_x-min_x)) - np.pi
        assert data.shape == (256,256,2), "DIMENSIONES DE LOS DATOS MAL! data: " + str(data.shape)
        return data
      else:
        raise Exception("Error cargando: ", path)
      
    def __getitem__(self, index):
        
        'Generate one batch of data'
        # Generate indexes of the batch
        pasos = self.batch_size
        indexes = self.indexes[index*pasos:(index+1)*pasos]

        # Find temporal list of paths to read
        list_paths_images_temp = [self.list_paths_images[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_paths_images_temp)

        X = self.resize(X)
        return X,[X[:,:,:,0], X[:,:,:,1], X[:,:,:,0], (X[:,:,:,1]+np.pi)/(2*np.pi)]#, X[:,:,:,0], X[:,:,:,1], X[:,:,:,0], X[:,:,:,1], X[:,:,:,0], X[:,:,:,1]]
        #return X,[X[:,:,:,0], X[:,:,:,1], X[:,:,:,0], X[:,:,:,1], X[:,:,:,0], X[:,:,:,1], X[:,:,:,0], X[:,:,:,1], X[:,:,:,0], X[:,:,:,1]]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_paths_images))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_paths_images_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, 256,256, self.n_channels))

        # Generate data
        for idx, path in enumerate(list_paths_images_temp):
            # Store sample
            x = self.read_mat(path)          
            X[idx, ...] = x
            
        
        return X
    
    def load_image(self, path):
        X = np.empty((1, 256,256, self.n_channels))
        X[0, ...] = self.read_mat(path)

        return X


class DataGeneratorMNIST(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, mnist, batch_size=128, dim=(256,256), n_channels=2,shuffle=False):
        'Initialization'
        #assert os.path.exists(path_images), "EL DIRECTORIO " + path_images + " NO EXISTE"

        self.mnist = mnist
        self.resize = tf.keras.layers.experimental.preprocessing.Resizing(*dim)
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dim = dim
        self.on_epoch_end()

    def __len__(self):
      'Denotes the number of batches per epoch'
      return (len(self.mnist)//self.batch_size)

 
    def __getitem__(self, index):
        
        'Generate one batch of data'
        # Generate indexes of the batch
        pasos = self.batch_size
        indexes = self.indexes[index*pasos:(index+1)*pasos]

        # Generate data
        X = self.mnist[indexes,...]

        X = self.resize(X)
        return X,[X[:,:,:,0], X[:,:,:,1], X[:,:,:,0], X[:,:,:,1], X[:,:,:,0], X[:,:,:,1], X[:,:,:,0], X[:,:,:,1], X[:,:,:,0], X[:,:,:,1]]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.mnist))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

