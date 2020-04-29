import Parser
import BranchNBound
import numpy as np
import pandas as pd

instance = Parser.TSPInstance('datasets/burma14.tsp')
instance.readData()
print(pd.DataFrame(data=instance.data))
# testData1 = np.array([
#     [np.inf, 27, 43, 16, 30, 26],
#     [7, np.inf, 16, 1, 30, 25],
#     [20, 13, np.inf, 35, 5, 0],
#     [21, 16, 25, np.inf, 18, 18],
#     [12, 46, 27, 48, np.inf, 5],
#     [23, 5, 5, 9, 5, np.inf]
# ])
visited, cost = BranchNBound.branchNbound(0, np.array(instance.data))
print(np.array(visited) + 1, cost)
