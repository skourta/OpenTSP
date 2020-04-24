import Parser
import BranchNBound
import numpy as np
import pandas as pd

instance = Parser.TSPInstance('datasets/bays29.tsp')
instance.readData()
# print(pd.DataFrame(data=instance.data))
testData1 = np.array([
    [np.inf, 10, 8, 9, 7],
    [10, np.inf, 10, 5, 6],
    [8, 10, np.inf, 8, 9],
    [9, 5, 8, np.inf, 6],
    [7, 6, 9, 6, np.inf]
])
visited, cost = BranchNBound.branchNbound(0, np.array(testData1))
print(np.array(visited) + 1, cost)
