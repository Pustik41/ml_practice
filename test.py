import pandas as pd
import numpy as np
import math
from statistics import mean

y = [1,3,5,7]
x = [2,4,6,8]
y = np.array(y)
x = np.array(x)


y_mean_line = [ mean(y) for i in y]

print(y_mean_line)
print(mean(y))
