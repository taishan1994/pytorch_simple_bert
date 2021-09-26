import numpy as np

x = np.array([[0.10786477, 0.56611762, 0.10557245], [0.4596513 , 0.13174377, 0.0]])

y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
print(y)