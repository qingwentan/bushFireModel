"""
Purpose:
    This script is designed to create an interactive web-based visualization platform
    for bushfire simulation outcomes, allowing users to explore the effects of various
    input parameters on bushfire simulation outputs. 
    Features of the web interface include viewing boxplots, zooming in/out, and saving the figures. 

Author:  Ying Zhu
Last Edited:  11 Sep, 2023
Usage:
    1. Install the necessary packages if haven't install:
        'pip install dash plotly'
    2. Run the command: 'python <path_to_app_visualisation.py>'
        Example: 'python3 src/app_visualisation.py'
"""

import dash
import plotly.express as px
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output


# Load all CSVs into a dictionary based on their strategy name
strategy_files = {
    "Goes to the Closest Fire": "data/output/curated/Goes to the closest fire_result.csv",
    "Goes to the Biggest Fire": "data/output/curated/Goes to the biggest fire_result.csv",
    "Parallel Attack": "data/output/curated/Parallel attack_result.csv",
    "Optimized Parallel Attack": "data/output/curated/Optimized Parallel attack_result.csv",
    "Optimized Closest": "data/output/curated/Optimized closest_result.csv",
    "Random Movements": "data/output/curated/Random movements_result.csv",
    "Indirect Attack": "data/output/curated/Indirect attack_result.csv",
}

data_dict = {strategy: pd.read_csv(filepath) for strategy, filepath in strategy_files.items()}

# Add a combined 'Total' dataframe
data_dict['Total'] = pd.concat(data_dict.values(), ignore_index=True)

# choices of strategies for dropdown
strategies = list(data_dict.keys())

# choices of inputs of simulation (x) for dropdown
inputs = ['truck_strategy', 'break_width', 'num_firetruck', 'max_speed', 'wind_dir', 
                 'steps_to_extinguish_1_cell', 'placed_on_edges']

# choices of outputs of simulation (y) for dropdown
outputs = ["Number_of_steps_to_ExtinguishFire", "Number_of_extinguished_firecell", 
                  "Number_of_burned_out_cell", "Number_of_cells_on_fire", 
                  "Maximum_growthrate_firecell_perstep",
                  "Maximum_extinguishedrate_firecell_perstep", 
                  "Count_healthy_trees", "Count_unhealthy_trees", "Percentage_damaged_burnedtrees"]

app = dash.Dash(__name__)
server = app.server
app.title = "Bushfire Visualiser"



app.layout = html.Div([

    # header 
    html.Div([
        html.H1("Interactive Visualisation of Bushfire Simulations",
                style={'padding': '30px', 'backgroundColor': '#f7f7f7',
                    'borderBottom': '2px solid #333', 'marginBottom': '20px', 'fontWeight': 'bold'})
    ], style={'textAlign': 'center'}),
    
    html.Div([
        # strategy dropdown
        html.Div([
            html.Label('Choose Strategy:'),
            dcc.Dropdown(
                id='strategy-dropdown',
                options=[{'label': s, 'value': s} for s in strategies],
                value='Total'  # default value
            ),
        ], style={'width': '30%', 'margin-left': '30px', 'display': 'inline-block'}),

        # input dropdown
        html.Div([
            html.Label('Choose X-axis (Simulation Input):'),
            dcc.Dropdown(
                id='input-dropdown',
                options=[{'label': i, 'value': i} for i in inputs],
                value='num_firetruck'  # default value
            ),
        ], style={'width': '30%', 'margin-left': '40px', 'display': 'inline-block'}),

        # output dropdown
        html.Div([
            html.Label('Choose Y-axis (Simulation Output):'),
            dcc.Dropdown(
                id='output-dropdown',
                options=[{'label': o, 'value': o} for o in outputs],
                value="Number_of_steps_to_ExtinguishFire"  # default value
            ),
        ], style={'width': '30%', 'margin-left': '40px', 'display': 'inline-block'})
    ]),
    
    dcc.Graph(id='boxplot-graph')
])

@app.callback(
    Output('boxplot-graph', 'figure'),
    [Input('strategy-dropdown', 'value'),
     Input('input-dropdown', 'value'),
     Input('output-dropdown', 'value')]
)


# a function to update the graph when any dropdown selection is changed
def update_graph(selected_strategy, selected_input, selected_output):

    # Check if any of the dropdown values is None
    if not selected_input or not selected_output or not selected_strategy:
        # render an empty figure and text reminder to select features
        return {
            'data': [],
            'layout': {
                'title': 'Select strategy, input and output to view visualisation.'
            }
        }
    
    # Select the appropriate dataframe from the dictionary
    filtered_data = data_dict[selected_strategy]
    # draw fig
    fig = px.box(filtered_data, x=selected_input, y=selected_output, height=600)
    # Dynamically determine the tick values based on unique values in the selected input column
    unique_tickvals = filtered_data[selected_input].unique()
    fig.update_xaxes(tickvals=unique_tickvals)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=3000)

