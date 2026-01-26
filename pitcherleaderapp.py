import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pybaseball import pitching_stats

data = pitching_stats(2025,2025, qual=162)

data["Age"].plot(kind = 'pie', figsize = (14,6))
plt.show()