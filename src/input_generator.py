"""
Purpose: This script is to generate various combination of input parameters, stored as multiple csv files.
         Each file contains different environment and agent settings for 1 fire strategy.
         The generated input files are stored at "ABM-master/data/input/".      

Author: Ying Zhu

Date: 18 Aug, 2023

Usage:
python <path_to_input_generator>

Example:
python src/input_generator.py
"""

import csv
from itertools import product

'''
The Forestfire model takes 16 arguments as follows:
        height, width: The size of the grid to model; 100 predefined
        density: What fraction of grid cells have a tree in them; float with 2 digits range from (0.01, 1)
        temperature: influences the number of spontaneous fires; int range from [0,100]
        truck_strategy: the tactic that firetrucks will adhere to; 7 strategies str
        river_width: in the case that a river is present, what is its width; int range from 0-10
        break_width: in the case that a pre-made fire break is present, what is its width; int range from 0-10
        random_fires: boolean indicating whether spontaneous fires are present; boolean 0/1
        vision: the distance fire fighting agents can look around them; 100 predifined 
        truck_max_speed: the max speed with which firetruck agents can move around (grid cells/step); int 0-30
        wind_strength: the speed with which the wind moves; int 0-80 km/h
        wind_dir: string specifying the direction of the fire; 8 directions str
        sparse_ratio: the fraction of the vegetation that is sparse instead of dense; float with 1 digit 0-1
        steps_to_extinguishment: number of steps it takes for firetrucks to extinguish a burning cell; int 1-6 !!! TODO: check the meaning
        placed_on_edges: indicates whether the firetrucks are placed randomly over the grid, or equispaced on the rim; boolean 0/1
'''

# pre-defined map grid size
height = [100]
width = [100]


# "Simulation results showed that when Ctot > 0.45 the fire covered the entire forest, but when Ctot ≤ 0.45 the fire did not spread." Ref. https://www.tandfonline.com/doi/full/10.1080/21580103.2016.1262793
density = [0.8]


# this temperture is only used to determine if a random spontenous fire will happen 
# the spontenous fire is disabled, so temperature takes default of 20
temperature = [20]


truck_strategy = ['Goes to the closest fire', 'Goes to the biggest fire', 'Random movements',
                  'Parallel attack',
                  'Optimized Parallel attack', 'Optimized closest', 'Indirect attack']

# river_width and break_width are both set to [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# river size will not exceed 200 meters if the map is 1000 meters if river exists. 
# 0, 2 
river_width = [0,2]

# A firebreak or double track is a gap in vegetation or other combustible material that acts as a barrier to (slow or) stop the progress of a bushfire or wildfire.
# so no fire will be grow in this width
# 0 or 1. 
break_width = [0,1]

# disabled
random_fires = [0]

# 10,30,50
num_firetruck = [10,30,50]

# pre-defined
vision = [100]

# max_speed is set to [15, 20, 25, 30]
# speed < 15 is ignored because based on my research, speed is as low as 16-32km/h. (?)
# seek professional advise from Prof. Trent
max_speed = [i for i in range(15, 31, 5)]

# wind_strength is set to [0, 10, 20, 30, 40, 50, 60, 70, 80]
# low 20, medium 50, high 80
wind_strength = [20,50,80]

# the effect of wind direction really depends on terrain. north-west, south-east, pick,4, perpendicular to each other 
wind_dir = ["\u2197 North/East", "\u2198 South/East", "\u2199 South/West", "\u2196 North/West"]

# sparse_ratio is set to [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
sparse_ratio = [0]

# low, medium, high rate (McCarthy 400m,300,200m. rate of extinguishnent)
# 4, 1, 0.5 hours
# 6, 2, 1, steps (the highest step we can set is 6)
steps_to_extinguishment = [1,2,6]

# 0,randomly placement at beginning more realistic 
placed_on_edges = [0,1]


parameters = [
    ("height", height),
    ("width", width),
    ("density", density),
    ("temperature", temperature),
    # "truck_strategy" will be set within the loop
    ("river_width", river_width),
    ("break_width", break_width),
    ("random_fires", random_fires),
    ("num_firetruck", num_firetruck),
    ("vision", vision),
    ("max_speed", max_speed),
    ("wind_strength", wind_strength),
    ("wind_dir", wind_dir),
    ("sparse_ratio", sparse_ratio),
    ("steps_to_extinguishment", steps_to_extinguishment),
    ("placed_on_edges", placed_on_edges)
]


# count the number of simulations in 1 strategy
n = 0
for var_name, var_value in parameters:
    temp = len(var_value)
    print(f"{var_name}: {var_value}")
    if n == 0:
        n = temp
    else: 
        n = n*temp
print("Number of Simulations in 1 strategy:", n)

# generate all combination of inputs by strategy and save as csv file 
for strategy in truck_strategy:
    parameters.insert(4,("truck_strategy", [strategy]))  # Set the current strategy
    keys = [param[0] for param in parameters]
    values = [param[1] for param in parameters]
    
    combinations = product(*values)

    # Save combinations of the current strategy to a CSV file
    with open(f"data/input/strategy_{strategy}.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(keys)
        for combo in combinations:
            writer.writerow(combo)

    # Remove the "truck_strategy" after writing the combinations for the current strategy
    parameters.pop(4)