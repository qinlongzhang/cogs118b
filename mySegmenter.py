import os
import cv2
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
import json
from Dao import saveToDB



#################################INCOMPLETE ############################################


class segmenterConfig(object):
	def __init__(self,filepath):
		self.load_config(filepath)
	def load_config(self,filepath):
		with open(filepath) as myfile:
			data = myfile.read()

		readData = json.loads(data)
		self.numGPU = readData["numGPU"]

		#for generator
		self.shuffle = readData["shuffle"]
		self.batchSize = readData["batchSize"]
		self.targetSize = tuple(readData["targetSize"])
		self.trainingAugAffine = readData["trainingAugAffine"]
		self.trainingAugChannel = readData["trainingAugChannel"]
		self.valAugAffine = readData["valAugAffine"]
		self.valAugChannel = readData["valAugChannel"]
		# for model
		self.earlyStopPatience = readData["earlyStopPatience"]
		self.reduceLr = readData["reduceLr"]
		self.reduceLrPa = readData["reduceLrPa"]
		self.reduceLrCD = readData["reduceLrCD"]
		self.learningRate = readData["learningRate"]
		self.numEpoch = readData["numEpoch"]
		self.baseWeight = readData["baseWeight"]
		self.epochResultFp = readData["epochResultFp"]
		if readData["optimizerString"].upper() = "SGD":
			self.optimizer = SGD(lr = self.learningRate)

		if readData["optimizerString"].upper() = "ADAM":
			self.optimizer = Adam(lr = self.learningRate)


		#for model DB
		self.dbInfo = {
		"host" = "127.0.0.1",
		"user" = "root",
		"passwd" = "Zql970502"
		}








class mySegmenter(object):
	model = None

	def __init__(self,configfilepath):
		self.model = None
		self.config = segmenterConfig(configfilepath)

	def train(self, logDir, trainingPath, validationPath):
		print("training examination filepath tra",trainingPath)
		print("training examination filepath val",validationPath)
		mydb = saveToDB(self.config.dbInfo)

		trainDataset = ImageSequence(csvFps = trainingPath,shuffle = self.config.shuffle,batchSize = self.config.batchSize,targetSize = self.config.targetSize,trainingAugAffine = self.config.trainingAugAffine,trainingAugChannel = self.config.trainingAugChannel)
		valDataset = ImageSequence(csvFps = validationPath, shuffle = self.config.shuffle,batchSize = self.config.batchSize,targetSize = self.config.targetSize,trainingAugAffine = self.config.valAugAffine,trainingAugChannel = self.config.valAugChannel)
		self.model = unet_2D(input_size = self.config.targetSize)
		early_stop = EarlyStopping(monitor = 'val_loss', patience = self.config.earlyStopPatience)
		reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = self.config.reduce_lr)
		csv_logger = CSVLogger(self.config.epochResultFp)
		checkpoint = ModelCheckpoint(os.path.join(logDir, "{epoch:03d}-{val_loss:.4f}.h5"))
		callbacks = [checkpoint,early_stop,reduce_lr,mydb]

		print("################start training###################")
		self.model.fit_generator(generator = trainDataset,epochs = 200, verbose =1, validation_data = valDataset,callbacks = callbacks,shuffle = False)

	def loadModel(self,modelPath):
		self.model = unet_2D(pretrained_weights = modelPath,input_size = (512,512,3))

	def predictImg(self,filePath):

		#img = cv2.imread(filePath)
		#img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		img = Image.open(filePath)
		img = np.array(img)
		img = img / 255
		#img = np.reshape(img,(img.shape + (1,)))
		img = np.reshape(img,((1,)+img.shape))
		return self.model.predict_on_batch(img)[0]

	def visualization(self,result,savePath,visible):
		result[np.where(result < 0.9)] = 0
		result[np.where(result >= 0.9)] = 255
		if visible:
			cv2.imshow("result",result)

		cv2.imwrite(savePath,result)







