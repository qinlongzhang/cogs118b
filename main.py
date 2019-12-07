from mySegmenter import mySegmenter

mode = 'p'
TRAIN_IMG_DIR = './images/train/img/'
TRAIN_MASK_DIR = './images/train/mask_mt/'
VAL_IMG_DIR = './images/val/img/'
VAL_MASK_DIR = './images/val/mask_mt/'

mySeg = mySegmenter()

if mode == 'train':
    mySeg.train('./log/', TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR)
else:
    mySeg.predictImg('./images/val/img/100.png', './log/result.h5', False, './output/prediction.jpg')
