import os
import random
import numpy as np
import pandas as pd
from keras.utils import Sequence
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image
from keras.utils import Sequence
#import imgaug as ia
#from imgaug import augmenters as iaa
# missing resize functianality
class ImageSequence(Sequence):
	# Note that classConfig stores list of pixels specifiying the class of target
	def __init__(self,imgDir,mskDir,shuffle,batchSize,classConfig,enableAugAffine,enableAugChannel):
		self.imgDir = imgDir
		self.mskDir = mskDir
		self.enableAugAffine = enableAugAffine
		self.enableAugChannel = enableAugChannel
		self.classConfig = classConfig
		self.imgList = []
		self.mskList = []
		self.shuffle = shuffle
		self.batchSize = batchSize
		self.on_epoch_end()
	# one hot dot helper convert the target to one Hot dot representation
	# parameter ndarray of size row*col*channel
	def oneHotDotHelper(self,img):
		imgTemp = img
		imgTemp = imgTemp.tolist()
		rowLen = img.shape[0]
		colLen = img.shape[1]

		for rdx in range(rowLen):
			for cdx in range(colLen):
				for idx in range(len(self.classConfig)):
					if imgTemp[rdx][cdx] == self.classConfig[idx]:
						conversion = np.zeros(len(self.classConfig))
						conversion[idx] = 1
						imgTemp[rdx][cdx] = conversion

		
		return np.asarray(imgTemp)

	def __len__(self):
		return int(np.floor(len(self.imgList)/self.batchSize))


	def __getitem__(self,idx):
		imgSubList = self.imgList[idx*self.batchSize:(idx+1)*self.batchSize]
		mskSubList = self.mskList[idx*self.batchSize:(idx+1)*self.batchSize]

		imgBatch,mskBatch = self.batchGenerator(imgSubList,mskSubList)

		return imgBatch,mskBatch
	def augmentationChannel(self):
		return iaa.SomeOf(1, [
			iaa.OneOf([
			iaa.Multiply((0.5, 1.5)),
			iaa.Add((-30, 60)),
			]),
			iaa.OneOf([
			iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
			iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.85, 1.5)),
			iaa.Emboss(alpha=(0.0, 1.0), strength=(0.9, 1.5)),
			]),
			iaa.OneOf([
				iaa.GaussianBlur(sigma=(0.0, 1.2)),
				iaa.AverageBlur(k=(2, 7)),
				iaa.MedianBlur(k=(3, 11)),
				iaa.Noop(),
				]
			)
		])

	def augmentationAffine(self):
		return iaa.SomeOf(1, [
				iaa.Affine(scale=0.5),
				iaa.Fliplr(0.5),
				iaa.Flipud(0.5),
			])

	def augment(self,imgBatch,mskBatch):
		ia.seed(random.randint(0,50))
		aug = self.augmentationChannel()
		aug1 = self.augmentationAffine().to_deterministic()
		imgBatch0 = imgBatch
		mskBatch0 = mskBatch
		if self.enableAugChannel and self.enableAugAffine:
			imgBatch0 = aug.augment_images(imgBatch)
			imgBatch0 = aug1.augment_images(imgBatch0)
			mskBatch0 = aug1.augment_images(mskBatch0)
		elif self.enableAugChannel == True and self.enableAugAffine == False:
			imgBatch0 = aug.augment_images(imgBatch0)
		elif self.enableAugChannel == False and self.enableAugAffine == True:
			imgBatch0 = aug1.augment_images(imgBatch0)
			mskBatch0 = aug1.augment_images(mskBatch0)
		else:
			imgBatch0 = imgBatch
			mskBatch0 = mskBatch


		return imgBatch0,mskBatch0

	def batchGenerator(self,imgSubList,mskSubList):
		imgBatch = []
		mskBatch = []
		for img,msk in zip(imgSubList,mskSubList):
			img = np.reshape(img,((1,)+img.shape))
			msk = np.reshape(msk,((1,)+msk.shape))
			img = img[0]
			msk = msk[0]
			if len(self.classConfig == 2):
				msk[np.where(msk!=0)] = 1
				msk = np.reshape(msk,msk.shape+(1,))
			else:
				msk = self.oneHotDotHelper(msk)
			img = img / 255
			imgBatch.append(img)
			mskBatch.append(msk)

		imgBatch = np.asarray(imgBatch)
		mskBatch = np.asarray(mskBatch)
		imgBatch,mskBatch = self.augment(imgBatch,mskBatch)

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




