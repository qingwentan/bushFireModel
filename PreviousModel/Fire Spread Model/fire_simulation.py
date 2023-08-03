from fire_spread_cella import FireSpreadModel
import matplotlib.pyplot as plt
import numpy as np

fmodel = FireSpreadModel(1, 5, 4, 0.7,0.4)

for i in range(10):
    fmodel.step()



print(fmodel.state)
