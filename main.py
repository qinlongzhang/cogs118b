from mySegmenter import mySegmenter

mySeg = mySegmenter()
mySeg.train('./log', './images/train/img', './images/val/img', './images/train/msk', './images/val/msk')
