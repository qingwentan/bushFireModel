# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

#Code to do a batch of simulations from a csv list of parameter sets and save the result into a file (run slow)
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import os.path as pth
import sys
import json
import csv
from time import time
from datetime import datetime, timedelta
from argparse import ArgumentParser

from SparkModel import get_param, SparkModel
from geostack.raster import Raster
from geostack.definitions import RasterNullValueType
from geostack.core import ProjectionParameters
from geostack.runner import runScript
from spark_ABM import ForestFire


logger = logging.getLogger("Spark")

# ABM's parameters
truck_strategy = 'Goes to the closest fire'
width = 512
height = 512
num_firetruck = 4
vision = 100
max_speed = 100
# in fact steps_to_extinguishment parameter do not use in ABM now
steps_to_extinguishment = 1
placed_on_edges = False

params = open("Simulations_input2.csv", 'r').readlines()[1:]

def matrixExtend(mtx):
    r_m = np.ones(mtx.shape[1])
    c_m = np.ones(512)
    for i in range(mtx.shape[1]):
        r_m[i] = np.NaN
    for i in range(512):
        c_m[i] = np.NaN
    while mtx.shape[0] < 512:
        mtx = np.row_stack((mtx, r_m))
    while mtx.shape[1] < 512:
        mtx = np.column_stack((mtx, c_m.T)) 

    #mtx = np.nan_to_num(mtx)
    return mtx


def sparkModel(cfgJson, comparisonFile=None):

    # Check json entries
    try:
        projection = get_param(cfgJson, "projection", (str,))
        jsonStartConditions = get_param(cfgJson, "startConditions", (dict,))
        resolutionMeters = get_param(cfgJson, 'resolutionMeters', (int, float,), checkGreater=0)
        levels = get_param(cfgJson, 'levels', (int, ), checkGreater=0, default=1)
        durationSeconds = get_param(cfgJson, "durationSeconds", (int, float,))
        timeMultiple = get_param(cfgJson, "timeMultiple", (int, float,))
        startDateISO8601 = get_param(cfgJson, "startDateISO8601", (str,), default='')
        initialisationModel = get_param(cfgJson, "initialisationModel", (str,))
        subModels = get_param(cfgJson, "subModels", (dict, str,))
        processingScript = get_param(cfgJson, "processingScript", (str,), default='')
        seriesCSVFile = get_param(cfgJson, "seriesCSVFile", (str,))
        seriesCSVData = get_param(cfgJson, "seriesCSVData", (str,))
        outputProjection = get_param(cfgJson, "outputProjection", (str,))
        inputLayers = get_param(cfgJson, "inputLayers", (list,))
        inputVectors = get_param(cfgJson, "inputVectors", (list,))
        outputLayers = get_param(cfgJson, "outputLayers", (list,))
        metLayers = get_param(cfgJson, "metLayers", (list,))
        metSpeedConversion = get_param(cfgJson, 'metSpeedConversion', (int, float,), default=1)
        jsonMask = get_param(cfgJson, "mask", (dict,))
        variables = get_param(cfgJson, "variables", (dict,))
        variablesCSVData = get_param(cfgJson, "variablesCSVData", (str,list,))
        boundaryPadding = get_param(cfgJson, "boundaryPadding", (int,), default=0)
        resultsFile = get_param(cfgJson, "resultsFile", (str,))

    except Exception as e:

        # Return error if any parameters are incorrect
        logger.error(f"Parameter error: {str(e)}")
        return -1
        logger.info(f"Geostack Spark solver ({datetime.now()})")
    
    printtofile = [["Strategy","Number of Truck","Max Speed","Vision","Placed on Edges","Truck Extinguish","Self Extinguish"]]
    
    for param in params:
        paramvalues = [i for i in param.rstrip().split(',')]
        truck_strategy = paramvalues[0]
        num_firetruck = int(paramvalues[3])
        vision = int(paramvalues[4])
        max_speed = int(paramvalues[5])
        # in fact steps_to_extinguishment parameter do not use in ABM now
        #steps_to_extinguishment = int(paramvalues[0])
        placed_on_edges = paramvalues[6]
        startTime = time()
        # Create Spark model
        ABM_model = ForestFire(
            height,
            width,
            truck_strategy,
            num_firetruck,
            vision,
            max_speed,
            steps_to_extinguishment,
            placed_on_edges
        )

        #trucks = [agent.pos for agent in ABM_model.schedule_FireTruck.agents]
        #print(trucks)
    
        spark_model = SparkModel()
        spark_model.configure(projection=projection,
                          resolutionMeters=resolutionMeters,
                          levels=levels,
                          durationSeconds=durationSeconds,
                          timeMultiple=timeMultiple,
                          startDateISO8601=startDateISO8601)

        # Check levels
        if levels > 1:
            logger.info(f"Using {levels} levels")

        # Set sub-models
        spark_model.set_submodels(subModels)

        # Set input layers
        if inputLayers and len(inputLayers) > 0:
            logger.info("Using input layers:")
        spark_model.set_input_layers(inputLayers=inputLayers)

        # Backwards compatibility for mask layer
        if jsonMask is not None:
            if inputVectors is None:
                inputVectors = []
            inputVectors.append({
                "name": "mask",
                "projection": "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
                "mapping": "distance",
                "data": jsonMask
            })

        # Set input vectors
        if inputVectors and len(inputVectors) > 0:
            logger.info("Using input vectors:")
        spark_model.set_input_vectors(inputVectors=inputVectors)

        # Set output layers
        spark_model.set_output_layers(outputLayers=outputLayers)

        # Get series data
        if seriesCSVData:
            spark_model.set_series_input(seriesCSVData=seriesCSVData)
        elif seriesCSVFile:
            spark_model.set_series_input(seriesCSVData=seriesCSVFile)

        # Set met layers
        if metLayers and len(metLayers) > 0:
            logger.info(f"Using met layers:")

            # Add wind speed conversion
            for item in metLayers:
                if 'type' in item and item['type'] == 'wind_magnitude':
                    item['conversion'] = metSpeedConversion

            # Set met input
            spark_model.set_met_input(metLayers=metLayers)

        # Parse variables
        if variables and len(variables) > 0:
            for name, value in variables.items():
                spark_model.set_variable(name, value)
        # parse variables csv file and set variables
        if variablesCSVData:
            spark_model.set_variable(variablesCSVData=variablesCSVData)

        # Parse models
        spark_model.set_initialisation_model(initialisationModel)
        spark_model.set_ros_models(rosModels=cfgJson)
        spark_model.set_update_models(updateModels=cfgJson)

        # Get start conditions
        spark_model.set_sources(sources=jsonStartConditions)

        # Initialise solver
        timer = time()
        logger.info("Initialising.. ")
        spark_model.initialise_solver()

        # Run solver
        logger.info(f"{time() - timer:f} s")
        timer = time()
        if timeMultiple is not None:
            nextReportTimeStep = np.ceil(durationSeconds*10.0/(100*timeMultiple))*timeMultiple
        else:
        # I think 15minutes one report is better, it can change 
            nextReportTimeStep = durationSeconds*10.0/400.0
        logger.info("Running.. ")

        # Output initial time
        if timeMultiple is not None:
            area = spark_model.solver_area * 0.0001
            if startDateISO8601 == '':
                currentEpochMilliSeconds = spark_model.solver.getEpochMilliseconds()
                logger.info(f"Time: {currentEpochMilliSeconds/1000} s; Area: {area:4.2f} ha; 0.00%")
            else:
                solverDateTime = spark_model.startDateTime + timedelta(seconds = spark_model.solver_time)
                solverDateTimeStr = solverDateTime.strftime("%Y-%m-%d %H:%M:%S%z")
                logger.info(f"Time: {solverDateTimeStr}; Area: {area:4.2f} ha; 0.00%")

        # Run model
    
        #proj = ProjectionParameters.from_proj4(projection)
        #difference = Raster(name = "difference")
        #difference.setProjectionParameters(proj)    
        #difference.init(nx = 600, ny = 600,          
        #             hx = resolutionMeters, hy = resolutionMeters)

        progressPercent = 2.5
        difference = np.zeros((512,512))
        truck_extinguish = 0 
        self_extinguish = 0 
    

        while spark_model.run_model(nextReportTimeStep):

            # Output log data
            area = spark_model.solver_area * 0.0001
            if startDateISO8601 == '':
                currentEpochMilliSeconds = spark_model.solver.getEpochMilliseconds()
                logger.info(f"Time: {currentEpochMilliSeconds/1000} s; Area: {area:4.2f} ha; {progressPercent:2.2f}%")
            else:
                solverDateTime = spark_model.startDateTime + timedelta(seconds = spark_model.solver_time)
                solverDateTimeStr = solverDateTime.strftime("%Y-%m-%d %H:%M:%S%z")
                logger.info(f"Time: {solverDateTimeStr}; Area: {area:4.2f} ha; {progressPercent:2.2f}%")

            #fig,axes = plt.subplots(1,2)
            #ax1 = axes[0]
            #ax2 = axes[1]
            #ax1.imshow(spark_model.get_output(name = "real_time_intensity").data,interpolation='none', origin='lower')
            #ax2.imshow(spark_model.get_output(name = "max_intensity").data,interpolation='none', origin='lower')
            #ax3.imshow(spark_model.get_arrival().data, interpolation='none', origin='lower')
            #plt.show()
            real_time_intensity = spark_model.get_output(name = "real_time_intensity")
            spark_data = real_time_intensity.data
            burn_area = len(spark_data[np.isnan(spark_data)==False])
            spark_data = matrixExtend(spark_data)
            all_area = spark_data.shape[0]*spark_data.shape[1]
            abm_input = spark_data - difference
        
            if progressPercent >= 5:
                ABM_model.step_intensity(abm_input)
                ABM_model.step()
                ABM_result = ABM_model.output_array()

                difference = spark_data - ABM_result
                self_extinguish = len(ABM_result[np.where(ABM_result == 0)])- (all_area - burn_area) - truck_extinguish
                #print(difference[np.where(difference > 1)])
                #ax2.imshow(ABM_result,interpolation='none', origin='lower')
            else:
                self_extinguish = len(abm_input[np.where(abm_input == 0)])
        
            truck_extinguish = ABM_model.count_extinguished_fires(ABM_model)
            print("Fire cells extinguished by truck:" + str(truck_extinguish))
            print("Fire spots extinguished by self:" + str(self_extinguish))
            #plt.show()
            progressPercent += 2.5
        
        printtofile.append([truck_strategy,num_firetruck,vision,max_speed,placed_on_edges,truck_extinguish,self_extinguish])
        
        logger.info(f"{time() - timer:f} s")
        timer = time()

        # Create results dictionary
        results = {}

        # Resize boundary
        if boundaryPadding is not None and boundaryPadding > 0:
            spark_model.resize_boundary(boundaryPadding)

        # Process output layers
        if processingScript is not None:
            spark_model.process_output(processingScript=processingScript)
            for name, value in spark_model.outputLayerReductions.items():
                results[name] = value

        logger.info("Writing.. ")

        # Write raster layers to tiff file
        spark_model.write_output_rasters(outputLayers=outputLayers,
                                     outputProjection=outputProjection)

        # Create and write isochrones
        outputIsochroneFile = get_param(cfgJson, "outputIsochroneJSONFile", (str,))
        if outputIsochroneFile is not None and outputIsochroneFile != "":
            outputIsochroneSeconds = get_param(cfgJson, "outputIsochroneTimeSeconds",
                                            (int, float), 0, 3600)
            outputIsochroneType = get_param(cfgJson, "outputIsochroneType",
                                            (str,), default='lines')
            spark_model.write_isochrones(output_interval=outputIsochroneSeconds,
                                        outfile=outputIsochroneFile,
                                        type = outputIsochroneType)

        # Write spot fires
        spark_model.write_spot_fires(outfile='./_out_spotfires.geojson') # TODO update name
        spark_model.get_spot_fire_distribution(write=True, outfile='./_out_spotfires.tif') # TODO update name

        # End
        logger.info(f"{time() - timer:f} s")

        # Comparison
        #if cfgJson.get("comparison", False):
        #    comparisonFile = get_param(cfgJson["comparison"], "source", (str,))
        #    comparisonOutputFile = get_param(cfgJson["comparison"], "destination", (str,))
        #    jaccard_score = spark_model.compare_output(inpFile=comparisonFile,
        #                                               outFile=comparisonOutputFile,
        #                                               method='jaccard')

            # Add to comparison set
        #    comparison = {}
        #    comparison['Jaccard'] = { 'value': jaccard_score, 'worst': 0, 'best': 1 }
        #    results['comparison'] = comparison

    # Write results
    if results:
        if resultsFile is not None:
            logger.info(f"Writing results to file: {resultsFile}")
            with open(resultsFile, "w") as rf:
                #rf.write(json.dumps(results, indent=4))
                writer = csv.writer(rf)
                for row in printtofile:
                    # write a row to the csv file
                    writer.writerow(row)
        else:
            logger.info("Results:")
            logger.info(results)

    logger.info(f"Finished in {time() - startTime:f} s")
    return 0

if __name__ == "__main__":

    # Parse command line arguments
    parser = ArgumentParser(description="Spark solver")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--configJSON", dest="configJSON", type=str,
        help="Spark json configuration")
    group.add_argument("--configFile", dest="configFile", type=str,
        help="Path to Spark json configuration file")
    parser.add_argument("--logToFile", dest="logToFile", action='store_true',
        help="Log output to file rather than console")
    args = parser.parse_args()

    # Set logging options
    if args.logToFile:
        logHandler = logging.FileHandler(f"spark_{datetime.today().strftime('%Y-%m-%d')}.log", 'a')
        logHandler.setLevel(logging.DEBUG)
        logFormatter = logging.Formatter('%(asctime)s; %(levelname)s: %(message)s')
        logHandler.setFormatter(logFormatter)
        logger.addHandler(logHandler)
        logger.setLevel(logging.DEBUG)
    else:
        logHandler = logging.StreamHandler()
        logHandler.setLevel(logging.INFO)
        logFormatter = logging.Formatter('%(message)s')
        logHandler.setFormatter(logFormatter)
        logger.addHandler(logHandler)
        logger.setLevel(logging.INFO)

    # Print usage and raise error if spark configuration is not provided
    if args.configJSON is None and args.configFile is None:
        parser.print_help()
        logger.error("Spark Configuration file is not provided")
        raise ValueError("Spark Configuration file is not provided")

    # Run Spark using configuration (json) as string
    if args.configJSON is not None:

        # Open configuration json file
        try:
            cfgJson = json.loads(args.configJSON)
        except ValueError as e:
            logger.error("Spark JSON configuration is not valid")
            raise ValueError("Spark JSON configuration is not valid")

        # Run
        if sparkModel(cfgJson) != 0:
            logger.error("Unable to run Spark model")
            raise RuntimeError("Unable to run Spark model")

    # Run Spark using configuration (json) file
    elif args.configFile is not None:
        if not pth.exists(args.configFile) or not pth.isfile(args.configFile):

            # Error if file cannot be found
            logger.error(f"Configuration file '{args.configFile}' not found")
            raise FileNotFoundError(f"Configuration file '{args.configFile}' not found")

        else:

            # Open configuration json file
            with open(args.configFile, 'r') as inp:

                # Parse json file
                cfgJson = json.load(inp)

                # Set working directory
                workingDir = pth.dirname(pth.abspath(args.configFile))
                os.chdir(workingDir)

                # Run
                if sparkModel(cfgJson) != 0:
                    logger.error("Unable to run Spark model")
                    raise RuntimeError("Unable to run Spark model")
