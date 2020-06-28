import Parser
import numpy as np
import pandas as pd
import AC as AC

instance = Parser.TSPInstance('datasets/kroA200.tsp')
instance.readData()
_colony_size = 5
_steps = 50
for i in [10, 50, 100]:
    print("Steps = {}, Colony Size = {}".format(i, 1000))
    acs = AC.AC(np.array(instance.data), mode="ACS", colony_size=i, steps=50)
    acs.run()
    print("\n", flush=True)
# acs.plot()
