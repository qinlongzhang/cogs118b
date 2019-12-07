from mySegmenter import mySegmenter

mode = 'p'

mySeg = mySegmenter()

if mode == 'train':
    mySeg.train('./log', './images/train/img/', './images/train/mask/', './images/val/img/', './images/val/mask/')
else:
    mySeg.predictImg('./images/train/img/0.jpg', './log/result.h5', False, './output/prediction.jpg')
