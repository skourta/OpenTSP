import Parser
import BranchNBound
import numpy as np
import pandas as pd
import AC as AC
import random

instance = Parser.TSPInstance('datasets/bays29.tsp')
instance.readData()
print(pd.DataFrame(data=instance.data))
testData1 = np.array([
    [np.inf, 27, 43, 16, 30, 26],
    [7, np.inf, 16, 1, 30, 25],
    [20, 13, np.inf, 35, 5, 1],
    [21, 16, 25, np.inf, 18, 18],
    [12, 46, 27, 48, np.inf, 5],
    [23, 5, 5, 9, 5, np.inf]
])
# visited, cost = BranchNBound.branchNbound(0, np.array(instance.data))
# print(np.array(visited) + 1, cost)

_colony_size = 5
_steps = 50
acs = AC.AC(np.array(instance.data), mode="ACS", colony_size=_colony_size, steps=_steps)
acs.run()
# acs.plot()
