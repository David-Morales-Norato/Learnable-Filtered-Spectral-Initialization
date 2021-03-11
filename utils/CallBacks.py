import json
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np


class SaveResults_CB(tf.keras.callbacks.Callback):

  def __init__(self, results_folder, generator, type_model, ext='.png'):
    if (not os.path.exists(results_folder)):
      print("Creando directorio de resultados en : ", results_folder)
      os.makedirs(results_folder)
    self.path_json_file = os.path.join(results_folder, "json_file.json")
    self.generator = generator

    self.data = {}

    self.Image_loss =  os.path.join(results_folder, type_model+"_loss_hist"+ ext)
    self.Image_loss_initial_abs =  os.path.join(results_folder, type_model+"_loss_initial_abs_hist"+ ext)
    self.Image_loss_initial_ang =  os.path.join(results_folder, type_model+"_loss_initial_ang_hist"+ ext)
    self.Image_loss_abs =  os.path.join(results_folder, type_model+"_loss_abs_hist"+ ext)
    self.Image_loss_angle =  os.path.join(results_folder, type_model+"_loss_angle_hist"+ ext)
    self.Image_Reference_Path = os.path.join(results_folder, type_model+"_X_real"+ ext)
    self.Image_Reconstruction_Path = os.path.join(results_folder, type_model+"_predictions"+ ext)
    self.Image_Initialization_Path = os.path.join(results_folder, type_model+"_initialization"+ ext)
    self.Image_SSIM_final_hist = os.path.join(results_folder, type_model+"_ssim_hist_final"+ ext)
    self.Image_SSIM_init_hist =os.path.join(results_folder, type_model+"_ssim_hist_initial"+ ext)
    self.Image_PSNR_final_hist = os.path.join(results_folder, type_model+"_psnr_hist_final"+ ext)
    self.Image_PSNR_init_hist =os.path.join(results_folder, type_model+"_psnr_hist_initial"+ ext)
    self.pesos_path = os.path.join(results_folder, type_model+"_last_weights.h5")

    self.data["images"] = {}
    self.data["images"]["Image_loss"] = self.Image_loss
    self.data["images"]["Image_Reference_Path"] = self.Image_Reference_Path
    self.data["images"]["Image_Reconstruction_Path"] = self.Image_Reconstruction_Path

    self.data["vars"] = {}
    self.data["vars"]["epoch"] = None
    self.data["vars"]["loss"] = None
    self.data["vars"]["val_loss"] = None

    self.data["errors"] = []   
    self.data["var_activity_bot"] = "epoch"

    self.ssim_abs_hist = []
    self.ssim_angle_hist = []
    self.psnr_abs_hist = []
    self.psnr_phase_hist = []

    
    self.ssim_init_abs_hist = []
    self.ssim_init_angle_hist = []
    self.psnr_init_abs_hist = []
    self.psnr_init_phase_hist = []

    self.loss_hist = []
    self.val_loss_hist = []
    self.loss_init_abs  = []
    self.loss_init_anlge  = []
    self.val_init_loss_abs  = []
    self.val_init_loss_angle  = []
    self.loss_abs_hist = []
    self.val_loss_abs_hist = []
    self.loss_angle_hist = []
    self.val_loss_angle_hist = []

    with open(self.path_json_file,'w') as f:
        json.dump(self.data, f, indent=4)

  def write_var(self, var, value):
    self.data["vars"][var] = value


  def save_figure(self, X, name, epoch = -1):
    fig, axs = plt.subplots(1, 2,figsize=(10,10))
    a = axs[0].imshow(X[0][0,:,:]); plt.sca(axs[0]); plt.yticks([]); plt.xticks([])
    fig.colorbar(a, ax=axs[0],fraction=0.046)
    a = axs[1].imshow(X[1][0,:,:])
    fig.colorbar(a, ax=axs[1],fraction=0.046); plt.sca(axs[1]); plt.yticks([]); plt.xticks([])
    fig.suptitle('Epoch: ' + str(epoch))
    fig.savefig(name)
    plt.close(fig)


  def save_plot(self, X, labels, name, epoch=-1):
    for l,n in zip(X,labels):
      plt.plot(l, label = n)
    plt.legend()
    plt.title('Epoch: ' + str(epoch))
    plt.savefig(name)
    plt.close()

  def get_metrics(self, real, pred):
    pass
    
  def on_epoch_end(self, epoch, logs=None):
    
    keys = list(logs.keys())
    print(keys)
    # try:
    self.write_var("epoch", epoch)
    self.write_var("loss", logs['loss'])
    self.write_var("val_loss", logs['val_loss'])
    
    X_f, X_real = self.generator.__getitem__(2)
    predictions = self.model(X_f)

    pred_init_abs = predictions[0]
    pred_init_ang = predictions[1]
    pred_abs = predictions[-2]
    pred_ang = predictions[-1]

    real_abs = X_real[0]
    real_ang = X_real[1]

    # SSIM
    ssim_init_abs = tf.image.ssim(real_abs, pred_init_abs, max_val=1)
    ssim_init_phase = tf.image.ssim(real_ang, pred_init_ang, max_val=1)
    ssim_abs = tf.image.ssim(real_abs, pred_abs, max_val=1)
    ssim_phase = tf.image.ssim(real_ang, pred_ang, max_val=1)

    # PSNR
    psnr_init_abs = tf.image.psnr(real_abs, pred_init_abs, max_val=1)
    psnr_init_phase = tf.image.psnr(real_ang, pred_init_ang, max_val=1)
    psnr_abs = tf.image.psnr(real_abs, pred_abs, max_val=1)
    psnr_phase = tf.image.psnr(real_ang, pred_ang, max_val=1)

    # LOSSES TRAIN
    self.loss_hist.append(logs['loss'])
    self.loss_init_abs.append(logs[keys[1]])
    self.loss_init_anlge.append(logs[keys[2]])
    self.loss_abs_hist.append(logs[keys[3]])
    self.loss_angle_hist.append(logs[keys[4]])

    # LOSSES VALIDATION
    self.val_loss_hist.append(logs['val_loss'])
    self.val_init_loss_abs.append(logs[keys[5]])
    self.val_init_loss_angle.append(logs[keys[6]])
    self.val_loss_abs_hist.append(logs[keys[7]])
    self.val_loss_angle_hist.append(logs[keys[8]])

    # SSIM
    self.ssim_init_abs_hist.append(ssim_init_abs)
    self.ssim_init_angle_hist.append(ssim_init_phase)
    self.ssim_abs_hist.append(ssim_abs)
    self.ssim_angle_hist.append(ssim_phase)


    # PSNR
    self.psnr_init_abs_hist.append(psnr_init_abs)
    self.psnr_init_phase_hist.append(psnr_init_phase)
    self.psnr_abs_hist.append(psnr_abs)
    self.psnr_phase_hist.append(psnr_phase)


    self.save_plot([self.loss_hist,self.val_loss_hist], ["loss train", "loss validation"], self.Image_loss, epoch=epoch)
    self.save_plot([self.loss_init_abs,self.val_init_loss_abs], ["loss init train", "loss init validation"], self.Image_loss_initial_abs, epoch=epoch)
    self.save_plot([self.loss_init_anlge,self.val_init_loss_angle], ["loss init train", "loss init validation"], self.Image_loss_initial_ang, epoch=epoch)
    self.save_plot([self.loss_abs_hist,self.val_loss_abs_hist], ["loss abs train", "loss abs validation"], self.Image_loss_abs, epoch=epoch)
    self.save_plot([self.loss_angle_hist,self.val_loss_angle_hist], ["loss angle train", "loss angle validation"], self.Image_loss_angle, epoch=epoch)

    self.save_plot([self.ssim_abs_hist, self.ssim_angle_hist], ["Abs", "Angle"], self.Image_SSIM_final_hist, epoch=epoch)
    self.save_plot([self.psnr_abs_hist, self.psnr_phase_hist], ["Abs", "Angle"], self.Image_PSNR_final_hist, epoch=epoch)
    self.save_plot([self.ssim_init_abs_hist, self.ssim_init_angle_hist], ["Abs", "Angle"], self.Image_SSIM_init_hist, epoch=epoch)
    self.save_plot([self.psnr_init_abs_hist, self.psnr_init_phase_hist], ["Abs", "Angle"], self.Image_PSNR_init_hist, epoch=epoch)

    
    self.save_figure([real_abs, real_ang], self.Image_Reference_Path, epoch = epoch)
    self.save_figure([pred_init_abs, pred_init_ang], self.Image_Initialization_Path, epoch = epoch)
    self.save_figure([pred_abs, pred_ang], self.Image_Reconstruction_Path, epoch = epoch)

    with open(self.path_json_file,'w') as f:
      json.dump(self.data, f, indent=4)



class SaveResults_CBMNIST(tf.keras.callbacks.Callback):

  def __init__(self, results_folder, generator, type_model, ext='.png'):
    if (not os.path.exists(results_folder)):
      print("Creando directorio de resultados en : ", results_folder)
      os.makedirs(results_folder)
    self.path_json_file = os.path.join(results_folder, "json_file.json")
    self.generator = generator

    self.data = {}

    self.Image_loss =  os.path.join(results_folder, type_model+"_loss_hist"+ ext)
    self.Image_loss_initial_abs =  os.path.join(results_folder, type_model+"_loss_initial_abs_hist"+ ext)
    self.Image_loss_initial_ang =  os.path.join(results_folder, type_model+"_loss_initial_ang_hist"+ ext)
    self.Image_Reference_Path = os.path.join(results_folder, type_model+"_X_real"+ ext)
    self.Image_Initialization_Path = os.path.join(results_folder, type_model+"_initialization"+ ext)
    self.Image_SSIM_init_hist =os.path.join(results_folder, type_model+"_ssim_hist_initial"+ ext)
    self.Image_PSNR_init_hist =os.path.join(results_folder, type_model+"_psnr_hist_initial"+ ext)
    self.pesos_path = os.path.join(results_folder, type_model+"_last_weights.h5")
    self.Image_Reconstruction_Path = os.path.join(results_folder, type_model+"_predictions"+ ext)
    self.Image_SSIM_final_hist = os.path.join(results_folder, type_model+"_ssim_hist_final"+ ext)
    self.Image_PSNR_final_hist = os.path.join(results_folder, type_model+"_psnr_hist_final"+ ext)
    self.Image_loss_abs =  os.path.join(results_folder, type_model+"_loss_abs_hist"+ ext)
    self.Image_loss_angle =  os.path.join(results_folder, type_model+"_loss_angle_hist"+ ext)







    self.data["images"] = {}
    self.data["images"]["Image_loss"] = self.Image_loss

    self.data["vars"] = {}
    self.data["vars"]["epoch"] = None
    self.data["vars"]["loss"] = None
    self.data["vars"]["val_loss"] = None

    self.data["errors"] = []   
    self.data["var_activity_bot"] = "epoch"

    
    self.ssim_init_abs_hist = []
    self.ssim_init_angle_hist = []
    self.psnr_init_abs_hist = []
    self.psnr_init_phase_hist = []
    self.ssim_abs_hist = []
    self.ssim_angle_hist = []
    self.psnr_abs_hist = []
    self.psnr_phase_hist = []



    self.loss_hist = []
    self.val_loss_hist = []
    self.loss_init_abs  = []
    self.loss_init_anlge  = []
    self.val_init_loss_abs  = []
    self.val_init_loss_angle  = []

    self.loss_abs_hist = []
    self.val_loss_abs_hist = []
    self.loss_angle_hist = []
    self.val_loss_angle_hist = []





    with open(self.path_json_file,'w') as f:
        json.dump(self.data, f, indent=4)

  def write_var(self, var, value):
    self.data["vars"][var] = value


  def save_figure(self, X, name, epoch = -1):
    fig, axs = plt.subplots(1, 2,figsize=(10,10))
    a = axs[0].imshow(X[0][0,:,:]); plt.sca(axs[0]); plt.yticks([]); plt.xticks([])
    fig.colorbar(a, ax=axs[0],fraction=0.046)
    a = axs[1].imshow(X[1][0,:,:])
    fig.colorbar(a, ax=axs[1],fraction=0.046); plt.sca(axs[1]); plt.yticks([]); plt.xticks([])
    fig.suptitle('Epoch: ' + str(epoch))
    fig.savefig(name)
    plt.close(fig)


  def save_plot(self, X, labels, name, epoch=-1):
    for l,n in zip(X,labels):
      plt.plot(l, label = n)
    plt.legend()
    plt.title('Epoch: ' + str(epoch))
    plt.savefig(name)
    plt.close()

  def get_metrics(self, real, pred):
    pass
    
  def on_epoch_end(self, epoch, logs=None):
    #tf.print("LR: ", self.model.optimizer.lr)
    keys = list(logs.keys())
    # try:
    self.write_var("epoch", epoch)
    self.write_var("loss", logs['loss'])
    self.write_var("val_loss", logs['val_loss'])
    
    X_f,X_real = self.generator[2]
    predictions = self.model(X_f)

    pred_init_abs = predictions[0]
    pred_init_ang = predictions[1]
    pred_abs = predictions[-2]
    pred_ang = predictions[-1]  

    real_abs = X_real[0]
    real_ang = X_real[1]


    # SSIM
    ssim_init_abs = tf.image.ssim(real_abs, pred_init_abs, max_val=1)
    ssim_init_phase = tf.image.ssim(real_ang, pred_init_ang, max_val=2*np.pi)
    ssim_abs = tf.image.ssim(real_abs, pred_abs, max_val=1)
    ssim_phase = tf.image.ssim(real_ang, pred_ang, max_val=2*np.pi)

    # PSNR
    psnr_init_abs = tf.image.psnr(real_abs, pred_init_abs, max_val=1)
    psnr_init_phase = tf.image.psnr(real_ang, pred_init_ang, max_val=2*np.pi)
    psnr_abs = tf.image.psnr(real_abs, pred_abs, max_val=1)
    psnr_phase = tf.image.psnr(real_ang, pred_ang, max_val=2*np.pi)


    # SSIM
    self.ssim_init_abs_hist.append(ssim_init_abs)
    self.ssim_init_angle_hist.append(ssim_init_phase)
    self.ssim_abs_hist.append(ssim_abs)
    self.ssim_angle_hist.append(ssim_phase)


    # PSNR
    self.psnr_init_abs_hist.append(psnr_init_abs)
    self.psnr_init_phase_hist.append(psnr_init_phase)
    self.psnr_abs_hist.append(psnr_abs)
    self.psnr_phase_hist.append(psnr_phase)

    # LOSSES TRAIN
    self.loss_hist.append(logs['loss'])
    self.loss_init_abs.append(logs[keys[1]])
    self.loss_init_anlge.append(logs[keys[2]])
    self.loss_abs_hist.append(logs[keys[3]])
    self.loss_angle_hist.append(logs[keys[4]])

    # LOSSES VALIDATION
    self.val_loss_hist.append(logs['val_loss'])
    self.val_init_loss_abs.append(logs[keys[5]])
    self.val_init_loss_angle.append(logs[keys[6]])
    self.val_loss_abs_hist.append(logs[keys[7]])
    self.val_loss_angle_hist.append(logs[keys[8]])


    self.save_plot([self.loss_hist,self.val_loss_hist], ["loss train", "loss validation"], self.Image_loss, epoch=epoch)
    self.save_plot([self.loss_init_abs,self.val_init_loss_abs], ["loss init train", "loss init validation"], self.Image_loss_initial_abs, epoch=epoch)
    self.save_plot([self.loss_init_anlge,self.val_init_loss_angle], ["loss init train", "loss init validation"], self.Image_loss_initial_ang, epoch=epoch)



    self.save_plot([self.ssim_init_abs_hist, self.ssim_init_angle_hist], ["Abs", "Angle"], self.Image_SSIM_init_hist, epoch=epoch)
    self.save_plot([self.psnr_init_abs_hist, self.psnr_init_phase_hist], ["Abs", "Angle"], self.Image_PSNR_init_hist, epoch=epoch)
    self.save_plot([self.ssim_abs_hist, self.ssim_angle_hist], ["Abs", "Angle"], self.Image_SSIM_final_hist, epoch=epoch)
    self.save_plot([self.psnr_abs_hist, self.psnr_phase_hist], ["Abs", "Angle"], self.Image_PSNR_final_hist, epoch=epoch)
    
    self.save_figure([real_abs, real_ang], self.Image_Reference_Path, epoch = epoch)
    self.save_figure([pred_init_abs, pred_init_ang], self.Image_Initialization_Path, epoch = epoch)
    self.save_figure([pred_abs, pred_ang], self.Image_Reconstruction_Path, epoch = epoch)
    with open(self.path_json_file,'w') as f:
      json.dump(self.data, f, indent=4)

