from library import * 
import numpy as np 

model = loadModel(vmafProxy, 'model')

deg = np.random.uniform(0, 1, (1, 3, 128, 128, 1))
ref = np.random.uniform(0, 1, (1, 3, 128, 128, 1))

score = model([ref, deg], training=False)

print(score)