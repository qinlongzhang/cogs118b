import numpy as np

classConfig = [[127, 63, 128], [70, 70, 70], [0, 0, 0]]

def parseResult(result):
    arr = np.argmax(result, axis=2).flatten()
    ret = []

    for ele in arr:
        ret.append(classConfig[ele])

    return np.array(ret).reshape((256, 256, 3))
