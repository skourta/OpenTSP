import Parser
import BranchNBound
import numpy as np
import pandas as pd
import AC as AC
import random

instance = Parser.TSPInstance('datasets/burma14.tsp')
instance.readData()
print(pd.DataFrame(data=instance.data))
testData1 = np.array([
    [np.inf, 10, 8, 9, 7],
    [10, np.inf, 10, 5, 6],
    [8, 10, np.inf, 8, 9],
    [9, 5, 8, np.inf, 6],
    [7, 6, 9, 6, np.inf]
])
visited, cost = BranchNBound.branchNbound(0, np.array(instance.data))
print(np.array(visited) + 1, cost)

_colony_size = 5
_steps = 50
acs = AC.AC(np.array(instance.data), colony_size=_colony_size, steps=_steps)
acs.run()
# acs.plot()