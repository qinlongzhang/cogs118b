import os
import random
import numpy as np
import pandas as pd
from keras.utils import Sequence
#import imgaug as ia
#from imgaug import augmenters as iaa
from PIL import Image
from keras.utils import Sequence
# missing resize functianality
class ImageSequence(Sequence):
	def __init__(self,imgDir,mskDir,shuffle,batchSize):
		self.imgDir = imgDir
		self.mskDir = mskDir
		self.imgList = []
		self.mskList = []
		self.shuffle = shuffle
		self.batchSize = batchSize
		self.on_epoch_end()


	def __len__(self):
		return int(np.floor(len(self.imgList)/self.batchSize))


	def __getitem__(self,idx):
		imgSubList = self.imgList[idx*self.batchSize:(idx+1)*self.batchSize]
		mskSubList = self.mskList[idx*self.batchSize:(idx+1)*self.batchSize]

		imgBatch,mskBatch = self.batchGenerator(imgSubList,mskSubList)

		return imgBatch,mskBatch

	def batchGenerator(self,imgSubList,mskSubList):
		imgBatch = []
		mskBatch = []
		for img,msk in zip(imgSubList,mskSubList):
			img = np.reshape(img,((1,)+img.shape))
			msk = np.reshape(msk,((1,)+msk.shape))
			img = img[0]
			msk = msk[0]
			msk[np.where(msk!=0)] = 1
			msk = np.reshape(msk,msk.shape+(1,))
			img = img / 255
			imgBatch.append(img)
			mskBatch.append(msk)

		imgBatch = np.asarray(imgBatch)
		mskBatch = np.asarray(mskBatch)

		return imgBatch,mskBatch

	def on_epoch_end(self):
		self.imgList = []
		self.mskList = []
		imgDirList = os.listdir(self.imgDir)
		for eachImg in imgDirList:
			self.imgList.append(np.array(Image.open(self.imgDir+eachImg)))
			self.mskList.append(np.array(Image.open(self.mskDir+eachImg)))

		if self.shuffle == True:
			seed = np.random.randint(50)
			random.Random(seed).shuffle(self.imgList)
			random.Random(seed).shuffle(self.mskList)




