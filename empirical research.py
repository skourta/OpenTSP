import Parser
import numpy as np
import pandas as pd
import AC as AC

instance = Parser.TSPInstance('datasets/berlin52.tsp')
instance.readData()
_colony_size = 5
_steps = 50
# for i in [10, 50, 100, 250, 500, 1000]:
#     print("Steps = {}, Colony Size = {}".format(100, 200))
#     acs = AC.AC(np.array(instance.data), mode="ACS", colony_size=200, steps=100)
#     acs.run()
#     print("\n", flush=True)
# for i in ["ACS", "Elitist", "MinMax"]:
#     print("Variation: {}, Steps = {}, Colony Size = {}".format(i, 100, 1000))
#     acs = AC.AC(np.array(instance.data), mode=i, colony_size=1000, steps=100)
#     acs.run()
#     print("\n", flush=True)

# acs.plot(Variatio

acs = AC.AC(np.array(instance.data), mode="ACS", colony_size=200, steps=100)
acs.run()
print("\n", flush=True)