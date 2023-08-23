'''
Forest Fire Simulation Runner
Author: Ying Zhu, Aanchal
Date: August 21, 2023

This script is designed to simulate forest fires using a set of parameters from an input CSV file. The main functionalities include:
- Reading parameters for the ForestFire model from an input CSV.
- Iteratively running the forest fire simulation based on each row of the CSV, treating them as separate configurations.
- Organizing simulation outputs into folders by truck strategy and simulation count.
- Saving model and agent results to their respective CSV files within the organized folders.

Usage:
python <path_to_simulation_runner> <path_to_input_csv>
(Note. Input CSV file path has multiple space so "" are requried.)

Example:
python src/simulation_runner.py "data/input/strategy_Goes to the biggest fire.csv"
'''


import time
import csv
import sys
import os
import pandas as pd
from main_model import ForestFire


def create_folder(path):
    """Create a folder at the specified path if not exist."""
    try:
        os.makedirs(path)
        print(f"Folder created at {path}")
    except FileExistsError:
        print(f"Folder already exists at {path}")
    except Exception as e:
        print(f"Error creating folder: {e}")

# Start the timer
start_time = time.time()

# Check if the CSV path is provided
if len(sys.argv) < 2:
    print("Please provide the path to the input CSV file.")
    sys.exit()

# get the input csv path
input_path = sys.argv[1]


# Open the CSV file and read row by row
with open(input_path, 'r') as file:
    # Create a DictReader object
    csv_reader = csv.DictReader(file)
    # index of current simulation
    count = 0 
    for row in csv_reader:
        # update index
        count += 1
        # get parameters
        height = int(row['height'])
        width = int(row['width'])
        density = float(row['density'])
        temperature = int(row['temperature'])
        truck_strategy = row['truck_strategy']
        river_width = int(row['river_width'])
        break_width = int(row['break_width'])
        random_fires = int(row['random_fires'])
        num_firetruck = int(row['num_firetruck'])
        vision = int(row['vision'])
        max_speed = int(row['max_speed'])
        wind_strength = int(row['wind_strength'])
        wind_dir = row['wind_dir']
        sparse_ratio = float(row['sparse_ratio'])
        steps_to_extinguishment = int(row['steps_to_extinguishment'])
        placed_on_edges = int(row['placed_on_edges'])
        # random_seed =?
        
        # initialise the fire model
        fire = ForestFire(
                height,
                width,
                density,
                temperature,
                truck_strategy,
                river_width, 
                break_width, 
                random_fires,
                num_firetruck,
                vision,
                max_speed,
                wind_strength,
                wind_dir,
                sparse_ratio,
                steps_to_extinguishment,
                placed_on_edges,
        )

        print(f"Simulation {count} Running...")

        fire.run_model()

        # create directory of the output file 
        create_folder(f'data/output/raw/{truck_strategy}/sim_{count}/')

        # save the input parameters
        input = pd.DataFrame([row])
        input.to_csv(f"data/output/raw/{truck_strategy}/sim_{count}/input.csv", index=False)

        # save model outputs
        results = fire.dc.get_model_vars_dataframe()
        results.to_csv(f"data/output/raw/{truck_strategy}/sim_{count}/model_result.csv", index=False)

        agent_variable = fire.dc.get_agent_vars_dataframe()
        agent_variable[0].to_csv(f"data/output/raw/{truck_strategy}/sim_{count}/agent_treeCell.csv")
        agent_variable[1].to_csv(f"data/output/raw/{truck_strategy}/sim_{count}/agent_firetruck.csv")


# End the timer
end_time = time.time()
print("ALL DONE!")

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"The simulation took {elapsed_time} seconds to run.")