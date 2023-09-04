'''
Forest Fire Simulation Runner
Author: Ying Zhu
Date: August 21, 2023

This script is designed to simulate forest fires using a set of parameters from an input CSV file. The main functionalities include:
- Reading parameters for the ForestFire model from an input CSV.
- Iteratively running the forest fire simulation based on each row of the CSV, treating them as separate configurations.
- Organizing simulation outputs into folders by truck strategy and simulation count.
- Saving model and agent results to their respective CSV files within the organized folders.

Usage:
python <path_to_simulation_runner> <path_to_input_csv>
(Note. Input CSV file path has multiple space so "" are requried.)

Examples:
python src/simulation_runner.py "data/input/strategy_Goes to the biggest fire.csv"
python src/simulation_runner.py "data/input/strategy_Indirect attack.csv"
'''


import time
import csv
import sys
import os
import pandas as pd
from main_model import ForestFire


def create_folder(path):
    """
        Create a folder at the specified path if not exist.
        message can be abled by uncommenting printing
    """
    try:
        os.makedirs(path)
        # print(f"Folder created at {path}")
    except FileExistsError:
        # print(f"Folder already exists at {path}")
        pass
    except Exception as e:
        print(f"Error creating folder: {e}")


def simplify_wind_dir(value):
    """ 
    Simplify the 'wind_dir' value for more concise folder naming.
    This function maps verbose wind direction descriptions to their corresponding abbreviations.
    """
    replacements = {
        "\u2197 North/East": "NE",
        "\u2198 South/East": "SE",
        "\u2199 South/West": "SW",
        "\u2196 North/West": "NW"
    }
    return replacements.get(value, value)

# Start the timer
start_time = time.time()

# Check if the input CSV path is provided
if len(sys.argv) < 2:
    print("Please provide the path to the input CSV file.")
    sys.exit()

# get the input csv path
input_path = sys.argv[1]

# Set the number of replications for a given scenario.
# Based on project time constraints, 11 replications have been set as the limit.
# With this number, most outcome parameters achieve a 95% confidence rate within a 0.1 margin of error.
# TODO: Detailed explaination can be found in "qingwen's ipynb"
replicationN = 11


# Open the CSV file and count the number of rows
# that is the number of different senarios for 1 strategy
with open(input_path, 'r') as file:
    num_rows = sum(1 for line in file) - 1

# Open the CSV file, start simulations
with open(input_path, 'r') as file:
    # Create a DictReader object
    csv_reader = csv.DictReader(file)

    for index, row in enumerate(csv_reader, start=1):
        # print process to log
        print(f"Processing simulation {index}/{num_rows}...")
        
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
        
        # folder name with input parameters, used for saving results
        folder_name = f"break{break_width}_nTrucks{num_firetruck}_speed{max_speed}_windDir_{simplify_wind_dir(wind_dir)}_steps{steps_to_extinguishment}_edges{placed_on_edges}"

        # iterating 'replicationN' times for each row/senario
        for i in range(replicationN):
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
            this_folder_name = folder_name + (f'_rep{i+1}')

            fire.run_model()

            # create directory of the output file 
            create_folder(f'data/output/raw/{truck_strategy}/{this_folder_name}/')

            # save the input parameters
            input = pd.DataFrame([row])
            input.to_csv(f"data/output/raw/{truck_strategy}/{this_folder_name}/input.csv", index=False)

            # save model outputs
            results = fire.dc.get_model_vars_dataframe()
            results.to_csv(f"data/output/raw/{truck_strategy}/{this_folder_name}/model_result.csv", index=False)
            agent_variable = fire.dc.get_agent_vars_dataframe()
            agent_variable[0].to_csv(f"data/output/raw/{truck_strategy}/{this_folder_name}/agent_treeCell.csv")
            agent_variable[1].to_csv(f"data/output/raw/{truck_strategy}/{this_folder_name}/agent_firetruck.csv")


# End the timer
end_time = time.time()
print("ALL DONE!")

# Calculate the elapsed time
elapsed_time = end_time - start_time
# Get the whole number of hours
hours = elapsed_time // 3600
# Get the remainder when divided by 3600 and then determine the number of minutes
minutes = (elapsed_time % 3600) // 60
print(f"Elapsed time: {hours} hours and {minutes} minutes")
