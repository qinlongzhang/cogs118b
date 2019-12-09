import os
import cv2
import numpy as np
import keras.backend as K
from keras.models import load_model
from keras.callbacks import CSVLogger,EarlyStopping,ReduceLROnPlateau,TensorBoard,ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras.models import *
#import scipy.misc
#import matplotlib.pyplot as plt
from networks import unet_2D
from generator import ImageSequence
from PIL import Image
from networks import dice_coefficient
from networks import dice_coefficient_loss
from parseResult import parseResult

CLASS_CONFIG = [[127, 63, 128], [70, 70, 70], [0, 0, 0]]

class mySegmenter(object):
	model = None

	def __init__(self):
		self.model = None

	def train(self, logDir, trainImgDir,trainMskDir, valImgDir,valMskDir):
		trainDataset = ImageSequence(imgDir = trainImgDir,mskDir = trainMskDir,shuffle = False,batchSize = 2, classConfig = CLASS_CONFIG, enableAugAffine = False, enableAugChannel = False)
		valDataset = ImageSequence(imgDir = valImgDir,mskDir = valMskDir,shuffle = False,batchSize = 2, classConfig = CLASS_CONFIG, enableAugAffine = False, enableAugChannel = False)
		self.model = unet_2D()
		early_stop = EarlyStopping(monitor = 'val_loss', patience = 5)
		reduce_lr = ReduceLROnPlateau(monitor = 'val_loss')
		csv_logger = CSVLogger(/Users/dichongshuo/Desktop/result.xlsx)
		checkpoint = ModelCheckpoint(os.path.join(logDir, "{epoch:03d}-{val_loss:.4f}.h5"))
		callbacks = [csv_logger,checkpoint,early_stop,reduce_lr]

		print("################start training###################")
		self.model.fit_generator(generator = trainDataset,epochs = 10, verbose =1, validation_data = valDataset,callbacks = callbacks,shuffle = False)

	def predictImg(self,filePath,modelpath,visible,savePath):
		self.model = unet_2D(pretrained_weights = modelpath)
		#img = cv2.imread(filePath)
		#img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		img = Image.open(filePath)
		img = np.array(img)
		img = img / 255
		#img = np.reshape(img,(img.shape + (1,)))
		img = np.reshape(img,((1,)+img.shape))
		result = self.model.predict_on_batch(img)[0]
		#result[np.where(result <= 0.95)] = 0
		#result[np.where(result > 0.95)] = 255
		result = parseResult(result)
		if visible:
			cv2.imshow('hehe', result)
		cv2.imwrite(savePath,result)
		return result







