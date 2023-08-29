"""
Purpose:
    This script is to calculate and retrieve the statistics from the output csvs for each simulation.

Author: 
    Ying Zhu, Aanchal

Date: 
    28 Aug, 2023

Usage:
    1. Download raw data from Google Drive.
    2. Place the "raw" folder under "data/test/", i.e., "data/test/raw".
    3. Run the command: 'python <path_to_get_statistics.py>'
       Example: 'python src/get_statistics.py'

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
        - Calculated as (8000*0.8) - Count of unhealthy Tree.
    2. Count of unhealthy Tree.
    3. Average damaged percentage for overall burned trees: 
        - Calculated as the average 'life bar' for unhealthy trees.
    4. Check if the burning rate is consistently 20 (verified programmatically, not visually).
"""

import os
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


# the content under the zip folder- "raw" folder - should be placed under data/test/.
# so the directory looks like "data/test/raw"
# and the output statistic will be saved in "data/test/curated"
base_path = "data/test/raw"
strategies = [dir_name for dir_name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, dir_name))]
print(strategies)

for strategy in strategies:
    # Initialize an empty list to store dictionaries with statistics values
    statistics_data = []
    strategy_path = os.path.join(base_path, strategy)
    sim_dirs = [sim_dir for sim_dir in os.listdir(strategy_path) if os.path.isdir(os.path.join(strategy_path, sim_dir)) and "sim_" in sim_dir]
    for sim_dir in sim_dirs:
        model_result_path = os.path.join(strategy_path, sim_dir, "model_result.csv")
        treeCell_path = os.path.join(strategy_path, sim_dir, "agent_treeCell.csv")
        if os.path.exists(model_result_path):
            model_result = pd.read_csv(model_result_path)
            #print(model_result)
            # Do whatever you need with the file
            # Count the rows in the DataFrame
            total_rows = model_result.shape[0]
            number_of_steps = total_rows
            #print(f"Number of steps in {model_result_path}: {number_of_steps}")
            # Extract and print the value of "Extinguished" column from the last row
            last_extinguished_value = model_result["Extinguished"].iloc[-1]
            #print(f"Value of 'Extinguished' in last row: {last_extinguished_value}")
            # Extract and print the value of 'Burned Out' column from the last row
            last_burned_out_value = model_result["Burned Out"].iloc[-1]
            #print(f"Value of 'Burned Out' in last row: {last_burned_out_value}")
             # Extract and print the value of Number of cells that were on fire
            fine_value = model_result["Fine"].iloc[-1]
            onFire=8000-fine_value
            #print(f"Number of cells that were on fire: {onFire}")
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

            #print("Maximum Fire Rate per Cell:", max_fire_rate)
            
            # Calculate absolute differences for the Extinguished firecell perstep

            extinguish_firecell_perstep = model_result['Extinguished'].diff().abs().max()
            #print("Extinguish Rate:", extinguish_firecell_perstep)

            # Find and print the overall maximum absolute differences
            maximum_growthrate_firecell = max(max_abs_diff_above) if max_abs_diff_above else 0
            
            #print(f"Overall maximum absolute difference above: {maximum_growthrate_firecell}")
           

        if os.path.exists(treeCell_path):
            treeCell = pd.read_csv(treeCell_path)

            #print(treeCell)
            # Do whatever you need with the file
            # Get the last 8000 rows
            last_8000_rows = treeCell.tail(8000)
            # Count rows where "Life bar" is not 100
            count_non_100 = last_8000_rows[last_8000_rows["Life bar"] != 100].shape[0]
            count_healthy_trees= 8000*0.8-(count_non_100)
            count_nonhealthy_trees= count_non_100
            #print("Number of healthy trees:", count_healthy_trees)
            #print("Number of unhealthy trees:", count_nonhealthy_trees)

            # Calculate the percentage of damaged tree
            percentage_damaged = (count_non_100 / 8000) * 100

            #print("Percentage of damaged trees:", percentage_damaged)


        create_folder("data/test/curated")
        output_path = os.path.join("data/test/curated", strategy)
        print(output_path)
        # create directory of the output file 
        create_folder(output_path)
        # Create the output directory for the current simulation
        #output_sim_dir = os.path.join(output_path, sim_dir)
        #create_folder(output_sim_dir)
    
        # "statistics" will be the name of final dataframe which has 9 columns and 1000 rows.
        statistics_data.append({"Number_of_steps_to_ExtinguishFire": number_of_steps,
            "Number_of_extinguished_firecell": last_extinguished_value,
            "Number_of_burned_out_cell": last_burned_out_value,
            "Number_of_cells_on_fire": onFire,
            "Maximum_growthrate_firecell_perstep": max_fire_rate,
            "Maximum_extinguishedrate_firecell_perstep": extinguish_firecell_perstep,
            #"Minimum_extinguishedrate_firecell_perstep": minimum_extinguishedrate_firecell,
            "Count_healthy_trees": count_healthy_trees,
            "Count_unhealthy_trees": count_nonhealthy_trees,
            "Percentage_damaged_burnedtrees": percentage_damaged
        })
        # Create the statistics DataFrame from the list of dictionaries
        statistics = pd.DataFrame(statistics_data)

        # Create the output file path for the current simulation
        output_file = os.path.join(output_path, "statistics.csv")

        # Save the statistics DataFrame to the output file
        statistics.to_csv(output_file, index=False)
       