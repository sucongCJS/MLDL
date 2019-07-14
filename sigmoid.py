import numpy as np
import matplotlib.pyplot as plt 

# 值域(0,1)
def sigmoid(t):
    return 1 / (1 + np.exp(-t))

x = np.linspace(-10, 10, 500) # 从-10到10有500个点
y = sigmoid(x)

plt.plot(x,y)
plt.show()