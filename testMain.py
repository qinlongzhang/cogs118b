import os
import cv2
import math
#import pydicom
import glob
import numpy as np
import pandas as pd
from skimage.transform import resize
from sklearn.metrics import roc_auc_score, mean_absolute_error
from keras.optimizers import Adam, SGD
import unittest
import json
import cv2
from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa
import math
import random
import json
#import tool
#obj test
class obj(object):
	def __init__(self):
		self.land = "myLand"
		self.owner = "stark"

class callObj(object):
	def __init__(self):
		self.thisObj = obj()

	def function(self):
		print(self.thisObj.land)

callObj = callObj()
callObj.function()
"""
img = Image.open("C:/Users/fact/Desktop/0 (1).tif")
resize_image(image, target_size)
"""
"""
val1 = True
val2 = False
if val1 and val2:
	print("double true")

if val1 and val2 == False:
	print("not double true")
	"""
"""
def listShuffle():
	df = pd.read_excel("C:/Users/fact/Desktop/testUnet/train.xlsx")
	imgList = df.iloc[:,0].tolist()
	mskList = df.iloc[:,1].tolist()
	#print(indexList[0])
	seed = np.random.randint(50)
	random.Random(seed).shuffle(imgList)
	random.Random(seed).shuffle(mskList)

	counter = 0
	while counter <= 20:
		print(imgList[counter])
		print(mskList[counter])
		counter = counter+1

listShuffle()
"""
#print(len(indexList))
#print(indexList[0])
#print(indexList[1])
"""
idxList = indexList[0:4]

for item in idxList:
	print(item)
	"""
"""
def augmentationAffine():
	return iaa.SomeOf(3,[
		iaa.Affine(rotate = -90),
		iaa.Affine(rotate = 90),
		iaa.Affine(rotate = 180),
		iaa.Fliplr(1),
		iaa.Flipud(1),
		])
def augmentationChannel():
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
			iaa.EdgeDetect(alpha=(0.0,1.0)),
			iaa.Noop(),
			]
			)
		])

def augment(imgBatch,mskBatch):
	ia.seed(random.randint(0,50))
	aug = augmentationChannel()
	aug1 = augmentationAffine().to_deterministic()
	imgBatch0 = aug.augment_images(imgBatch)
	mskBatch0 = mskBatch
	imgBatch0 = aug1.augment_images(imgBatch0)
	mskBatch0 = aug1.augment_images(mskBatch0)
	return imgBatch0,mskBatch0

df = pd.read_excel("C:/Users/fact/Desktop/testUnet/trainOverfit.xlsx")
image_list = df.iloc[:,0].tolist()
msk_list = df.iloc[:,1].tolist()
for img,msk in zip(image_list,msk_list):
	im = Image.open(img)
	imarray = np.array(im)
	imarray = np.reshape(imarray,((1,)+imarray.shape))
	ms = Image.open(msk)
	msarray = np.array(ms)

	msarray = np.reshape(msarray,(1,)+msarray.shape)
	imarray,msarray = augment(imarray,msarray)
	imarray = imarray[0]
	msarray = msarray[0]
	processedImg = Image.fromarray(imarray)
	processedMsk = Image.fromarray(msarray)
	msarray = np.reshape(msarray,msarray.shape+(1,))
	print("msarray shape is",msarray.shape)
	processedImg.show()
	processedMsk.show()
	#Image.show(processedMsk)
"""
"""
for filename in glob.glob("C:/Users/fact/Desktop/augLabel/*.tif"):
	im = Image.open(filename)
	imarray = np.array(im)
	msk_list.append(imarray)
	Image.fromarray(img)
"""



"""
def testAugmentation():
	image_list = []

	for filename in glob.glob("C:/Users/fact/Desktop/aug/*.jpg"):
		im = Image.open(filename)
		imarray = np.array(im)
		image_list.append(imarray)

	image_list = np.asarray(image_list)
	iaaI = augmentationAffine()
	iaaG = augmentationChannel()
	imgBatch = iaaI.augment_images(image_list)
	#imgBatch = iaaG.augment_images(imgBatch)

	for idx in range(imgBatch.shape[0]):
		img = Image.fromarray(imgBatch[idx],'RGB')
		img.show()

	

testAugmentation()

array = np.array([[0,0,10,0],[0,3,0,5],[0,10,0,7]])
array[np.where(array!=0)] = 1
print(array)
"""
"""
img = Image.open("C:/Users/fact/Desktop/aug/0 (1).tif")
img = np.array(img)
img
"""

#print(4 % 4)
"""
def yieldTest():
	i = 1

	while True:
		while i % 4 != 0:
			i = i+1

	yield i
	i = i+1



num = yieldTest()
print(num)
print(num.next())
"""
"""
def generator():
	arr1 = np.array([[1,2,3,4],[5,6,7,8]])
	arr2 = np.array([[4,5,6,7],[8,9,20,32]])
	msk1 = np.zeros((2,4))
	msk2 = np.zeros((2,4))
	imgList = []
	mskList = []
	imgList.append(arr1)
	imgList.append(arr2)
	mskList.append(msk1)
	mskList.append(msk2)

	for (img,msk) in zip(imgList,mskList):
		yield (img,msk)

result = generator()
print(next(result))
print(next(result))
"""
#arr = np.array([[5,6,7,8,9],[10,11,13,14,15]])
#arr = np.reshape(arr,arr.shape + (1,))
#print(arr[0][4][0])
"""
img = Image.open("C:/Users/fact/Desktop/0 (1).tif")
img = np.array(img)
img = Image.fromarray(img)
img.show()

img.save("C:/Users/fact/Desktop/0 (1)mirror.tif")
"""
#img = np.reshape(img, (img.shape + (1,)))
#img = np.reshape(img,((1,) + img.shape))
#print(img[0])
#df = pd.read_excel("C:/Users/fact/Desktop/testUnet/train.xlsx")
#print("filepath is what","C:/Users/fact/Desktop/testUnet/train.xlsx")
#imgFps = df.iloc[:,0].tolist()
#mskFps = df.iloc[:,1].tolist()
#print(len(imgFps))
#print(len(mskFps))
"""
li = []
arr1 = np.array([[1,2,3],[4,5,6]])
arr2 = np.array([[0,0,0],[1,3,4]])
li.append(arr1)
li.append(arr2)
li = np.asarray(li)
print(li)

print(li[0])
print(li[1])
"""
"""
array = np.array([[0,0,10,0],[0,3,0,5],[0,10,0,7]])
temp = np.zeros((3,4,2))
for row in range(array.shape[0]):
	for col in range(array.shape[1]):
		if array[row][col] != 0:
			temp[row][col] = [1,0]
		else:
			temp[row][col] = [0,1]

print(temp) 
"""
"""
img = Image.open("C:/Users/fact/Desktop/0 (1).tif")
img = np.array(img)

rowIdx = 0
colIdx = 0

for row in range(img.shape[0]):
	for col in range(img.shape[1]):
		if (img[row][col] == 255):
			rowIdx = row
			colIdx = col 
			break


print(rowIdx)
print(colIdx)
"""
#ii = cv2.imread("E:/Segmentation/img/0 (620)jpg.tif")
#gray_image = cv2.cvtColor(ii,cv2.COLOR_BGR2GRAY)
#print(type(gray_image))
#print(gray_image.shape)
#cv2.imshow("image",gray_image)
#cv2.waitKey(0)
#cv2.imwrite("C:/Users/fact/Desktop/gray0.tif",gray_image)
"""
gray_image = gray_image / 255
print(gray_image)
largest = gray_image[0][0]
for row in range(gray_image.shape[0]):
	for col in range(gray_image.shape[1]):
		if gray_image[row][col] > largest:
			largest = gray_image[row][col]

print(largest)
gray_image = np.reshape(gray_image,(gray_image.shape + (1,)))
gray_image = np.reshape(gray_image,((1,)+gray_image.shape))
print(gray_image.shape)
"""