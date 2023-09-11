"""
Purpose:
    This script compiles simulation outcomes, extracts statistical measures from individual
    simulation output CSVs, and consolidates the results across all 11 iterations. 
    The final output is a single CSV for one strategy, comprising 577 rows, where each row 
    presents both input parameters and corresponding output statistics.

Author: 
    Ying Zhu, Aanchal

Last Edited: 
    11 Sep, 2023

Usage:
    1. Download folder named as Strategy name from Google Drive: Data/output/raw/.
    2. Place the strategy folder under "data/output/raw/", i.e., "data/output/raw/Parallel attack".
    3. Run the command: 'python <path_to_get_statistics.py> "strategy_folder_name"'
       Example: 'python src/get_statistics.py "Parallel attack"'

Statistics Retrieved:

From "Model_result.csv":
    1. Number of steps/Time to extinguish fire: 
        - Equals the number of steps(rows) excluding the header.
    2. Number of extinguish firecell: 
        - Value from the last row under 'extinguish' column.
    3. Number of burned out cell: 
        - Value from the last row under 'Burned out' column.
    4. Number of cells that were on fire: 
        - Calculated as (8000 - last row of 'Fine') OR (Number of extinguish firecell + Number of burned out cell).
    5. Maximum Growth rate of firecell per step: 
        - Calculated from the 'On Fire' column as the maximum difference between any 2 adjacent cells.
    6. Maximum Extinguish rate of fire cell per step: 
        - Calculated from the 'Extinguish' point as the maximum difference between any 2 adjacent 'On Fire' cells.
    7. Minimum Extinguish rate of fire per step: 
        - Calculated similarly to point 6 but takes the minimum value.

From "agent_treeCell.csv":
    1. Count of Healthy Tree: 
        - Calculated as (8000) - Count of unhealthy Tree.
    2. Count of unhealthy Tree.
    3. Average damaged percentage for overall burned trees: 
        - Calculated as the average 'life bar' for unhealthy trees.
    4. Check if the burning rate is consistently 20 (verified programmatically, not visually).
"""

import os
import sys
import time 
import pandas as pd


def create_folder(path):
    """Create a folder at the specified path if not exist."""
    try:
        os.makedirs(path)
        print(f"Folder created at {path}")
    except FileExistsError:
        pass
    except Exception as e:
        print(f"Error creating folder: {e}")

# Check if the input strategy folder name is provided
if len(sys.argv) < 2:
    print("Please provide the strategy name.")
    sys.exit()

# get the input strategy name
strategy = sys.argv[1]

'''
If using get_statistic.py on test_1k_runs:
1. comment out the user input sys, line 60-66

2. uncomment the following line
strategies = [dir_name for dir_name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, dir_name))]

3. add an extra for loop as the outer side of the loop as follows:
    'for strategy in strategies: 
        for sim_dir in sim_dirs:
            ...
    '
'''


# Start the timer
start_time = time.time()

# Initiaise the result dataframe with column names
input_columns = ['truck_strategy', 'break_width', 'num_firetruck', 'max_speed', 'wind_dir', 
                 'steps_to_extinguishment', 'placed_on_edges']
output_columns = ["Number_of_steps_to_ExtinguishFire", "Number_of_extinguished_firecell", 
                  "Number_of_burned_out_cell", "Number_of_cells_on_fire", 
                  "Maximum_growthrate_firecell_perstep",
                  "Maximum_extinguishedrate_firecell_perstep", 
                  "Count_healthy_trees", "Count_unhealthy_trees", "Percentage_damaged_burnedtrees"]
columns = input_columns + output_columns
result = pd.DataFrame(columns=columns)

# identify paths
base_path = "data/output/raw"
strategy_path = os.path.join(base_path, strategy)
sim_dirs = [sim_dir for sim_dir in os.listdir(strategy_path) if os.path.isdir(os.path.join(strategy_path, sim_dir))]
total_sims = len(sim_dirs)
# iterate through every simulation folder, get its input.csv and outputs from output csvs
for index, sim_dir in enumerate(sim_dirs):
    # print process to log
    print(f"Processing: {index}/{total_sims}")

    # get paths of csvs outputs from simulation folder
    model_result_path = os.path.join(strategy_path, sim_dir, "model_result.csv")
    treeCell_path = os.path.join(strategy_path, sim_dir, "agent_treeCell.csv")
    input_path = os.path.join(strategy_path, sim_dir, "input.csv")

    # get statistics from model_result.csv
    if os.path.exists(model_result_path):
        model_result = pd.read_csv(model_result_path)
        # Count the rows in the DataFrame
        total_rows = model_result.shape[0]
        number_of_steps = total_rows
        # Extract and print the value of "Extinguished" column from the last row
        last_extinguished_value = model_result["Extinguished"].iloc[-1]
        # Extract and print the value of 'Burned Out' column from the last row
        last_burned_out_value = model_result["Burned Out"].iloc[-1]
        # Extract and print the value of Number of cells that were on fire
        fine_value = model_result["Fine"].iloc[-1]
        onFire = 8000 - fine_value
        # Initialize lists to store max absolute differences above and below
        max_abs_diff_above = []
        max_abs_diff_below = []
        # Find the index of the row with the maximum 'On Fire' value
        max_on_fire_index = model_result["On Fire"].idxmax()
        # Initialize maximum fire rate to zero
        max_fire_rate = 0
        # Initialize a list to store fire rates
        fire_rates = []
        # Iterate through the DataFrame starting from the second row
        for i in range(1, len(model_result)):
            fire_rate = (model_result.loc[i , 'On Fire'] + model_result.loc[i , 'Extinguished']) - (model_result.loc[i-1, 'Extinguished'] + model_result.loc[i-1, 'On Fire'])
            fire_rates.append(fire_rate)
            # Find the maximum fire rate from the list
            max_fire_rate = max(fire_rates)
        # Calculate absolute differences for the Extinguished firecell perstep
        extinguish_firecell_perstep = model_result['Extinguished'].diff().abs().max()
        # Find and print the overall maximum absolute differences
        maximum_growthrate_firecell = max(max_abs_diff_above) if max_abs_diff_above else 0
        
    # get statistics from treeCell.csv
    if os.path.exists(treeCell_path):
        treeCell = pd.read_csv(treeCell_path)
        # Get the last 8000 rows
        # since tree_density is always 0.8, and height=100, and width=100. so 0.8*10000 = 8000
        last_8000_rows = treeCell.tail(8000)
        # Count rows where "Life bar" is not 100
        count_non_100 = last_8000_rows[last_8000_rows["Life bar"] != 100].shape[0]
        count_healthy_trees= 8000*0.8-(count_non_100)
        count_nonhealthy_trees= count_non_100
        # Calculate the percentage of damaged tree
        percentage_damaged = count_non_100 / 8000


    # combine outputs from model_result and treeCell as a dictionary
    output_dict = {
        "Number_of_steps_to_ExtinguishFire": number_of_steps,
        "Number_of_extinguished_firecell": last_extinguished_value,
        "Number_of_burned_out_cell": last_burned_out_value,
        "Number_of_cells_on_fire": onFire,
        "Maximum_growthrate_firecell_perstep": max_fire_rate,
        "Maximum_extinguishedrate_firecell_perstep": extinguish_firecell_perstep,
        "Count_healthy_trees": count_healthy_trees,
        "Count_unhealthy_trees": count_nonhealthy_trees,
        "Percentage_damaged_burnedtrees": percentage_damaged * 100
        #"Percentage_of_extinguished_firecell": last_extinguished_value/10000,
        #"Percentage_of_burned_out_cell": last_burned_out_value/10000,
        #"Percentage_of_cells_on_fire": onFire/10000,
        #"Minimum_extinguishedrate_firecell_perstep": minimum_extinguishedrate_firecell,
    }

    # get the input parameters as dictionary
    if os.path.exists(input_path):
        input = pd.read_csv(input_path)
        # Drop columns where inputs are the constants
        columns_to_drop = ['height', 'width', 'density', 'temperature', 'river_width', 'random_fires',
                           'vision', 'wind_strength', 'sparse_ratio']
        input = input.drop(columns=columns_to_drop)
        # Convert the input DataFrame which is only 1 row to a dictionary
        input_dict = input.iloc[0].to_dict()


    # combine inputs and output as one row, and append it to the final result dataframe
    combined_row = {**input_dict, **output_dict}
    result.loc[len(result)]=combined_row

# After done with all simulations, now group by the results for same sets of input parameters,
# get the average result for 11 replications 
result = result.groupby(input_columns)[output_columns].mean().reset_index()

# Rename 'steps_to_extinguishment' for input paramters to 'steps_to_extinguish_1_cell'
# to avoid confusion with the "Number_of_steps_to_ExtinguishFire" in the output
df = result.rename(columns={'steps_to_extinguishment': 'steps_to_extinguish_1_cell'})

# create directory of the output file 
create_folder("data/output/curated")

# Save the result DataFrame as a csv to the directory 
# the directory is 'data/output/curated/strategy_name_result.csv'
output_path = os.path.join("data/output/curated", f"{strategy}_result.csv")
result.to_csv(output_path, index=False)

# End the timer
end_time = time.time()
print("ALL DONE!")
# Calculate the elapsed time in hours, minutes, seconds
elapsed_time = end_time - start_time
hours = elapsed_time // 3600
minutes = (elapsed_time % 3600) // 60
seconds = elapsed_time % 60
print(f"Elapsed time: {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds")