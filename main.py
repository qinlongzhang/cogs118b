from mySegmenter import mySegmenter

mode = 'predict'

mySeg = mySegmenter()

if mode == 'train':
    mySeg.train('./log', './images/train/img/', './images/train/mask/', './images/val/img/', './images/val/mask/')
else:
    mySeg.predictImg('./images/train/img/0.jpg', './log/011--0.7317.h5', False, './output/prediction.jpg')
