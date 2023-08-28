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
base_path = "data/output/raw"
strategies = [dir_name for dir_name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, dir_name))]
print(strategies)
for strategy in strategies:
    strategy_path = os.path.join(base_path, strategy)
    sim_dirs = [sim_dir for sim_dir in os.listdir(strategy_path) if os.path.isdir(os.path.join(strategy_path, sim_dir)) and "sim_" in sim_dir]
    for sim_dir in sim_dirs:
        model_result_path = os.path.join(strategy_path, sim_dir, "model_result.csv")
        treeCell_path = os.path.join(strategy_path, sim_dir, "agent_treeCell.csv")
        if os.path.exists(model_result_path):
            model_result = pd.read_csv(model_result_path)
            print(model_result)
            # Do whatever you need with the file

        if os.path.exists(treeCell_path):
            treeCell = pd.read_csv(treeCell_path)
            # print(treeCell)
            # Do whatever you need with the file


    create_folder("data/test/curated")
    output_path = os.path.join("data/test/curated", strategy)
    print(output_path)
    # create directory of the output file 
    create_folder(output_path)

    # "statistics" will be the name of final dataframe which has 12 columns and 1000 rows.
    # statistics.to_csv(output_path)