# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:27:30 2022

@author: zengy
"""

import json
import numpy as np

from geostack.raster import Raster
from geostack.runner import runScript
from SparkModel import SparkModel
# from spark_ABM import ForestFire
import pandas as pd

# truck_strategy = 'Goes to the closest fire'
width = 1000
height = 1000
# num_firetruck = 30
vision = 100
max_speed = 100
steps_to_extinguishment = 2
placed_on_edges = False

# Create Spark model
spark_model = SparkModel()

# ABM_model = ForestFire(
#     height,
#     width,
#     truck_strategy,
#     num_firetruck,
#     vision,
#     max_speed,
#     steps_to_extinguishment,
#     placed_on_edges
# )

#trucks = [agent.pos for agent in ABM_model.schedule_FireTruck.agents]
#print(trucks)

# Configure Spark model
timeMultiple = 28800
projStr = "+proj=lcc +lat_1=-36 +lat_2=-38 +lat_0=-37 +lon_0=145 +x_0=2500000 +y_0=2500000 +ellps=GRS80"
spark_model.configure(projection=projStr,
                      resolutionMeters=30.0,
                      durationSeconds=3600*8,
                      timeMultiple=timeMultiple,
                      startDateISO8601="2020-01-01T08:00:00Z")

# Set start conditions
startConditions = {
    "features": [
        {
            "geometry": {
                "coordinates": [
                    144.318,
                    -37.429
                ],
                "type": "Point"
            },
            "properties": {
                "radius": 100
            },
            "type": "Feature"
        }
    ],
    "type": "FeatureCollection"
}
spark_model.set_sources(sources=json.dumps(startConditions))

# Create barrier raster
name = "barrier"
barrier = Raster(name = name)
spark_model.outputRasters[name] = barrier

# Set rate-of-spread models
rosModels = {
    "rateOfSpreadModels" : {
        "1" : """
        // Define an arbitrary elliptical length to breadth ratio.
         REAL LBR = 3.0;

        // Determine coefficient for backing and flanking rank of spread using elliptical equations
        REAL cc = sqrt(1.0-pow(LBR, -2.0));
        REAL cb = (1.0-cc)/(1.0+cc);
        REAL a_LBR = 0.5*(cb+1.0);
        REAL cf = a_LBR/LBR;
        
        REAL f = 0.5*(1.0+cb);
        REAL g = 0.5*(1.0-cb);
        REAL h = cf;

        // Set the speed around the fire perimeter
        REAL wdot = dot(normalize(wind_vector), advect_normal_vector);
        speed = g*wdot+sqrt(h*h+(f*f-h*h)*wdot*wdot);
"""
    }
}
spark_model.set_ros_models(rosModels=json.dumps(rosModels))

# Set update models
updateModels = {
    "updateModels" : {
        "1" : """
        // Stop fire if barrier values are defined        
        if (isValid_REAL(barrier)) {
            state = 0;
        }
"""
    }
}
scp = spark_model.set_update_models(updateModels=json.dumps(updateModels))

# Set weather
spark_model.set_series_input("""
    date,wind_direction,wind_speed,temp,rel_hum
    2020-01-01T08:00:00Z,180,30,25,10
    2020-01-01T12:00:00Z,180,30,25,15
    2020-01-01T16:00:00Z,270,30,30,10
    2020-01-01T20:00:00Z,270,30,25,10
    """)
            
# Initialise solver
slover = spark_model.initialise_solver()


# Write barrier values
for i in range (100, 200):
    barrier.setCellValue(1, i, 200)

# Run model
import matplotlib.pyplot as plt
while spark_model.run_model(timeMultiple):

    # Grow barrier
    runScript("""
        if (isValid_REAL(barrier_N) || 
            isValid_REAL(barrier_S) || 
            isValid_REAL(barrier_E) || 
            isValid_REAL(barrier_W)) {
                barrier = 1.0;
        }
    """, [barrier])

    # Output image
    plt.imshow(spark_model.get_arrival().data, interpolation='none', origin='lower')
    plt.show()
    spark_result = spark_model.get_arrival().data
    final_result = pd.DataFrame(spark_result)
    final_result.to_csv("F_result.csv")

    # ABM_model.step_intensity(spark_result)
    # ABM_model.step()
    # trucks = [agent.pos for agent in ABM_model.schedule_FireTruck.agents]
    # print(trucks)
    # ABM_result = ABM_model.output_array()
    #print(ABM_model.count_extinguished_fires(ABM_model))
    # plt.imshow(ABM_result, interpolation='none', origin='lower')
    # plt.show()
    
    #plt.imshow(np.ma.masked_where(barrier.data == 0, barrier.data), interpolation='none', origin='lower')
    #print(spark_model.get_arrival().data[84][163])
    #x = np.where(~np.isnan(spark_result))[0]
    #print(x[0])
    #result = spark_model.get_output(name='barrier').data
    #print(result[130][150])
    #print(spark_model.get_classification().data)
    
    #print(np.shape(barrier.data))
