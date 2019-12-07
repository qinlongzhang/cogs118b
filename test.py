import os
import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image
import keras
from keras.engine import Input
from keras.utils import to_categorical
if __name__ == "__main__":
    a = np.array([1,2,3])
    b = Input(a)
    c = keras.activations.softmax(b)
    print(c)

        
    """
    a = np.array([[[1,2,3],[3,4,5]],[[3,4,5],[1,2,3]]])
    r = a.shape[0]
    c = a.shape[1]
    b = a.tolist()
    print(b[0][0])
    print(b[0][0] == [1,2,3])
    

    for rdx in range(r):
        for cdx in range(c):
            if b[rdx][cdx] == [1,2,3]:
                b[rdx][cdx] = np.array([1,0,0,1])
            else:
                b[rdx][cdx] = np.array([0,0,1,0])
    b = np.asarray(b)
    print(b.shape)
    print(b)      
    """
