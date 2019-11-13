import os
import cv2
import random
import numpy as np
import pandas as pd
from keras.utils import Sequence
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image
import tool
from keras.utils import Sequence
# missing resize functianality
class ImageSequence(Sequence):
	def __init__(self,csvFps,shuffle,batchSize,targetSize,trainingAugAffine,trainingAugChannel):
		self.csvFps = csvFps
		self.shuffle = shuffle
		self.batchSize = batchSize
		self.targetSize = targetSize
		self.trainingAugChannel = trainingAugChannel
		self.trainingAugAffine = trainingAugAffine
		self.on_epoch_end()
		#remove this in the real version this attempt to overfit 
		#self.imgList,self.mskList = self.overfit(overfitNum)

	#def augmentation(self,image):
	#def augmentationAffine():
	def augmentationChannel(self):
		return iaa.SomeOf(1,[
			iaa.OneOf([
			iaa.Multiply((0.5,1.5)),
			iaa.Add((-30,60)),
			]),
			iaa.OneOf([
			iaa.ContrastNormalization((1,1.5)),
			iaa.Sharpen(alpha = (0.0,1.0),lightness = (0.85,1.5)),
			iaa.Emboss(alpha = (0.0,1.0),strength = (0.9,1.5)),
				]),
			iaa.OneOf([
				iaa.GaussianBlur(sigma = (0.0,1.2)),
				iaa.Noop(),
				]
				)
			])

	def augmentationAffine(self):
		return iaa.SomeOf(3,[
			iaa.Affine(rotate = -90),
			iaa.Affine(rotate = 90),
			iaa.Affine(rotate = 180),
			iaa.Fliplr(1),
			iaa.Flipud(1),
			])


	def augment(self,imgBatch,mskBatch):
		ia.seed(random.randint(0,50))
		aug = self.augmentationChannel()
		aug1 = self.augmentationAffine().to_deterministic()
		imgBatch0 = imgBatch
		mskBatch0 = mskBatch
		if self.trainingAugChannel && self.trainingAugAffine:
			imgBatch0 = aug.augment_images(imgBatch)
			imgBatch0 = aug1.augment_images(imgBatch0)
			mskBatch0 = aug1.augment_images(mskBatch0)
		elif self.trainingAugChannel == True and self.trainingAugAffine == False:
			imgBatch0 = aug.augment_images(imgBatch0)
		elif self.trainingAugChannel == False and self.trainingAugAffine == True:
			imgBatch0 = aug1.augment_images(imgBatch0)
			mskBatch0 = aug1.augment_images(mskBatch0)
		else:
			imgBatch0 = imgBatch
			mskBatch0 = mskBatch


		return imgBatch0,mskBatch0

	def __len__(self):

		return int(np.floor(len(self.imgList)/self.batchSize))

	def __getitem__(self,idx):
		imgFpsList = self.imgList[idx*self.batchSize:(idx+1)*self.batchSize]
		mskFpsList = self.mskList[idx*self.batchSize:(idx+1)*self.batchSize]

		imgBatch,mskBatch = self.batchGenerator(imgFpsList,mskFpsList)

		return imgBatch,mskBatch

	def batchGenerator(self,imgFps,mskFps):
		imgBatch = []
		mskBatch = []
		for imgIdx,mskIdx in zip(imgFps,mskFps):
			img = Image.open(imgIdx)
			img = np.array(img)
			img = np.reshape(img,((1,)+img.shape))
			msk = Image.open(mskIdx)
			msk = np.array(msk)
			msk = np.reshape(msk,((1,)+msk.shape))
			img,msk = self.augment(img,msk)
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
		df = pd.read_excel(self.csvFps)
		tempImgList = df.iloc[:,0].tolist()
		tempMskList = df.iloc[:,1].tolist()

		self.imgList = tempImgList
		self.mskList = tempMskList

		if self.shuffle == True:
			seed = np.random.randint(50)
			random.Random(seed).shuffle(tempImgList)
			random.Random(seed).shuffle(tempMskList)
			self.imgList = tempImgList
			self.mskList = tempMskList





"""
	def overfit(self,number):
		df = pd.read_excel(self.csvFps)
		imgFps = df.iloc[:,0].tolist()
		mskFps = df.iloc[:,1].tolist()

		imgList = []
		mskList = []

		#img = cv2.imread(imgFps[0])
		#img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		#img = img / 255
		#img = np.reshape(img,(img.shape + (1,)))
		img = Image.open(imgFps[0])
		img = np.array(img)

		#img = img / 255
		msk = Image.open(mskFps[0])
		msk = np.array(msk)
		#msk[np.where(msk!=0)] = 1
		msk = np.reshape(msk,(msk.shape + (1,)))
		#ii = cv2.imread("E:/Segmentation/img/0 (620)jpg.tif")
		#gray_image = cv2.cvtColor(ii,cv2.COLOR_BGR2GRAY)

		for i in range(number):
			imgList.append(img)
			mskList.append(msk)

		return imgList,mskList


	def __len__(self):

		return int(np.floor(len(self.imgList)/self.batchSize))

	def __getitem__(self,idx):
		imgPartList = self.imgList[idx*self.batchSize:(idx+1)*self.batchSize]
		mskPartList = self.mskList[idx*self.batchSize:(idx+1)*self.batchSize]

		imgBatch,mskBatch = self.batchGenerator(imgPartList,mskPartList)

		return imgBatch,mskBatch

	def batchGenerator(self,imgList,mskList):
		imgBatch = []
		mskBatch = []

		for img,msk in zip(imgList,mskList):
			#img = self.augmentation(img)
			img = img / 255
			msk[np.where(msk!=0)] = 1
			imgBatch.append(img)
			mskBatch.append(msk)

		imgBatch = np.asarray(imgBatch)
		mskBatch = np.asarray(mskBatch)

		return imgBatch,mskBatch

"""


"""
def batchGenerator(imgFps,mskFps,batchSize,numSteps):
	#print("imgFps",imgFps)
	currentStep = 0
	imgId = 0
	imgBatchList = []
	mskBatchList = []
	while currentStep < numSteps:

		imgBatch,mskBatch = [],[]
		switchOn = True
		while imgId % batchSize != 0 or switchOn:
			switchOn = False
			rawMsk = Image.open(mskFps[imgId])
			rawMsk = np.array(rawMsk)
			rawMsk[np.where(rawMsk!=0)] = 1
			rawMsk = np.reshape(rawMsk,(rawMsk.shape + (1,)))
			rawImg = Image.open(imgFps[imgId])
			rawImg = np.array(rawImg)
			imgBatch.append(rawImg)
			mskBatch.append(rawMsk)
			imgId = imgId + 1
		imgBatch = np.asarray(imgBatch)
		mskBatch = np.asarray(mskBatch)
		imgBatchList.append(imgBatch)
		mskBatchList.append(mskBatch)
		currentStep = currentStep + 1

	return imgBatchList,mskBatchList
"""


"""
def segGenerator(filePathxlsx,batchSize,numSteps):
	df = pd.read_excel(filePathxlsx)
	print("filepath is what",filePathxlsx)
	imgFps = df.iloc[:,0].tolist()
	mskFps = df.iloc[:,1].tolist()
	imgBatchList,mskBatchList = batchGenerator(imgFps,mskFps,batchSize,numSteps)
	train_generator = zip(imgBatchList,mskBatchList)
	#print("calling batch")
	for (img,msk) in train_generator:
		yield(img,msk)

"""
# generate image for prediction


