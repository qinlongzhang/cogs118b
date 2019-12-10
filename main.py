from mySegmenter import mySegmenter

mode = 'train'

mySeg = mySegmenter()

if mode == 'train':
    mySeg.train('./log', './images/train/img/', './images/train/mask/', './images/val/img/', './images/val/mask/')
else:
    mySeg.predictImg('./images/train/img/0.jpg', './log/sgd.h5', False, './output/sgd.jpg')
