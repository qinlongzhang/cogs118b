import tensorflow as tf 
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from networks import *
from generator import *
from mySegmenter import *
if __name__ == "__main__":
	log_dir = "C:/Users/fact/Desktop/testUnet/log"
	trainingPath = "C:/Users/fact/Desktop/testUnet/train.xlsx"
	validationPath = "C:/Users/fact/Desktop/testUnet/val.xlsx"
	mySegmenter0 = mySegmenter()
	mySegmenter0.train(log_dir,trainingPath,validationPath)
	"""
	mySegmenter0.loadModel("C:/Users/fact/Desktop/testUnet/log/002--0.4772.h5")
	result = mySegmenter0.predictImg("C:/Users/fact/Desktop/0 (620)jpg.tif")

	largest = result[0][0]
	for row in range(result.shape[0]):
		for col in range(result.shape[1]):
			if largest < result[row][col]:
				largest = result[row][col]
	
	print(largest)
    
	mySegmenter0.visualization(result = result,savePath = "C:/Users/fact/Desktop/PredictImg0 (1).jpg" ,visible = True)
    """





