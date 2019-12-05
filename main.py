from mySegmenter import mySegmenter

mySeg = mySegmenter()
mySeg.train('./log', './images/train/img/', './images/train/mask/', './images/val/img/', './images/val/mask/')
