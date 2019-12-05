import os
import pandas as pd
import numpy as np
import keras.backend as K
from keras.models import load_model
from keras.callbacks import CSVLogger,EarlyStopping,ReduceLROnPlateau,TensorBoard,ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras.utils.training_utils import multi_gpu_model
from keras.models import *
#import scipy.misc
#import matplotlib.pyplot as plt
from networks import unet_2D
from generator import ImageSequence
import tool
from PIL import Image
from networks import dice_coefficient
from networks import dice_coefficient_loss
class mySegmenter(object):
	model = None

	def __init__(self,configfilepath):
		self.model = None

	def train(self, logDir, trainImgDir,trainMskDir, valImgDir,valMskDir):
		trainDataset = ImageSequence(imgDir = trainImgDir,mskDir = trainMskDir,shuffle = True,batchSize = 2)
		valDataset = ImageSequence(imgDir = valImgDir,mskDir = valMskDir,shuffle = True,batchSize = 2)
		self.model = unet_2D()
		early_stop = EarlyStopping(monitor = 'val_loss', patience = 5)
		reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5)
		#csv_logger = CSVLogger(self.config.epochResultFp)
		checkpoint = ModelCheckpoint(os.path.join(logDir, "{epoch:03d}-{val_loss:.4f}.h5"))
		callbacks = [checkpoint,early_stop,reduce_lr]

		print("################start training###################")
		self.model.fit_generator(generator = trainDataset,epochs = 200, verbose =1, validation_data = valDataset,callbacks = callbacks,shuffle = False)







