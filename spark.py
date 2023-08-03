import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geostack.raster import Raster
from geostack.runner import runScript
from SparkModel import SparkModel



# Create Spark model
spark_model = SparkModel()


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
    output_result = pd.DataFrame(spark_result)
    output_result.to_csv("data/rawData/output_result.csv")



