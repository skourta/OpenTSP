import Parser
import numpy as np
import pandas as pd
import AC as AC

instance = Parser.TSPInstance('datasets/bays29.tsp')
instance.readData()
# print(pd.DataFrame(data=instance.data))
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
for i in [10, 50, 100, 250, 500, 1000, 2000, 5000]:
    print("Steps = {}, Colony Size = {}".format(i, 1000))
    acs = AC.AC(np.array(instance.data), mode="ACS", colony_size = 1000, steps=i)
    acs.run()
    print("\n",flush=True)
# acs.plot()
