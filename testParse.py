from parseResult import parseResult
import numpy as np

a = np.random.random((256, 256, 3))

print(parseResult(a).shape)
