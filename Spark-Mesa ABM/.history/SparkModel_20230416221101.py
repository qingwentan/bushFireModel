# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 15:01:35 2022

@author: zengy
"""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import logging
import os
import os.path as pth
import sys
import re
import json
import io
import gdal
from time import time, sleep
from datetime import datetime
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Union, Optional, Iterable
from numbers import Integral, Real
from dataclasses import dataclass
import warnings
import numpy as np
from itertools import chain
from functools import partial
from math import floor

from geostack.solvers import LevelSet, Particle
from geostack.raster import Raster, RasterFile, RasterPtrList, RasterDimensions
from geostack.vector import Vector, BoundingBox, Coordinate
from geostack.core import ProjectionParameters, Variables
from geostack.io import geoJsonToVector, vectorToGeoJson, shapefileToVector, vectorToGeoWKT
from geostack.runner import stipple, runScript, runVectorScript
from geostack.series import Series
from geostack.definitions import SeriesInterpolationType, RasterInterpolationType
from geostack.definitions import GeometryType, ReductionType, VectorOrdering
from geostack.dataset import supported_libs

# get the logger
logger = logging.getLogger("Spark")

# Check parameter types and values
def get_param(cfgJson: Dict, param: str, type_options=None,
             checkGreater=None, default=None):

    # Return default if not found
    if param not in cfgJson:
        return default

    # Check type
    if type_options is not None:
        if not isinstance(type_options, tuple):
            logger.error("type_options should be tuple of datatype")
            raise TypeError("type_options should be tuple of datatype")

        for item in type_options:
            if item not in [int, float, str, dict, list]:
                logger.error(f"type_option {item.__name__} is not recognised")
                raise TypeError(f"type_option {item.__name__} is not recognised")

        if not isinstance(cfgJson[param], type_options):
            data_type = ','.join([f"{item.__name__}" for item in type_options])
            logger.error(f"{param} should be {data_type}")
            raise TypeError(f"{param} should be {data_type}")

    # Check value
    if checkGreater is not None:
        if cfgJson[param] <= checkGreater:
            logger.error(f"{param} should be greater than {checkGreater}")
            raise ValueError(f"{param} should be greater than {checkGreater}")

    return cfgJson[param]

# Parse multiple models from configuration
def parse_multiple_models(cfgJson: Dict,
                        prefix: str = None,
                        defaultModel: str = ""):
    # Check prefix
    if prefix is None:
        logger.error("prefix should not be None")
        raise ValueError("prefix should not be None")

    # Check key
    key_found = False
    for item in cfgJson:

        itemNoCase = item.lower()
        if itemNoCase.startswith(prefix.lower()):
            key_found = True
    if not key_found:
        return ""

    # Build script
    scriptStart = ""
    script = ""
    scriptEnd = ""
    for item in cfgJson:

        # Convert item to lowercase
        itemNoCase = item.lower()

        # Check key
        if itemNoCase.startswith(prefix.lower()):

            # Add model start
            if itemNoCase.endswith("modelscommonstart"):
                scriptStart = cfgJson[item] + "\n"

            # Add model body
            if itemNoCase.endswith("models"):

                # Parse union models, these treat the index as a bit field
                if 'unionmodels' in itemNoCase:

                    # Check values, union models can only have 22 bits
                    for key in cfgJson[item]:
                        if key.isnumeric() and (int(key) < 1 or int(key) > 23):
                            logger.error(f"Incorrect model id {key}, should be between 1-23")
                            raise KeyError(f"Incorrect model id {key}, should be between 1-23")

                    # Add model
                    for key in cfgJson[item]:
                        if key.isnumeric():
                            script += "if (_class_lo & %d) {\n %s }\n" % (
                                1 << int(key), cfgJson[item][key])
                # Parse indexed models, these use a separate index for each model
                else:

                    # Add model
                    script += "switch (_class_lo) {\n"
                    for key in cfgJson[item]:
                        if key.isnumeric():
                            script += "case %s: {\n %s} break;\n" % (key, cfgJson[item][key])
                    script += "default: { %s } break;\n}\n" % (defaultModel)

            # Add model end
            if itemNoCase.endswith("modelscommonend"):
                scriptEnd = cfgJson[item] + "\n"

    return scriptStart + script + scriptEnd

def parse_variables_file(variablesData: str, delimiter: str= '\n'):
    # Create output dictionary
    out = {}

    # Check for file
    if isinstance(variablesData, str):
        if delimiter not in variablesData:

            # Open csv file
            fp = open(variablesData, 'r')
        else:
            fp = variablesData.split(delimiter)
            fp = [item for item in fp if len(item) > 0]

        if isinstance(fp, io.TextIOWrapper):
            # Read headers
            line = fp.readline()
        elif isinstance(fp, list):
            line, fp = fp[0], fp[1:]

        if not line:
            logger.error(f"Header in csv file {variablesData} is empty.")
            raise RuntimeError(f"Header in csv file {variablesData} is empty.")
        headers = line.split(',')
        headers = [header.strip() for header in headers]
        if len(headers) == 0:
            logger.error(f"csv file {variablesData} has no headers.")
            raise RuntimeError(f"csv file {variablesData} has no headers.")

        # Create dictionary, skip first column
        for i, name in enumerate(headers, 1):
            out[name] = []

        # Parse rows
        for line in fp:

            # Read items
            items = line.split(',')

            # Add to series
            for i in range(len(items)):
                f = np.NAN
                try:
                    f = float(items[i].strip())
                except ValueError:
                    pass
                out[headers[i]].append(f)

        if isinstance(fp, io.TextIOWrapper):
            fp.close()
    else:
        logger.error(f"File {variablesData} cannot be located.")
        raise FileNotFoundError(f"File {variablesData} cannot be located.")
    return out


def parse_series_file(seriesData: str, delimiter:str = '\n'):
    """parse input file with time series data.

    Parameters
    ----------
    seriesData : str
        path and name of the CSV file, or string of CSV data.

    Returns
    -------
    Series
        an instance of Series object with time series data.

    Raises
    ------
    RuntimeError
        "header in csv file is empty"
    RuntimeError
        "csv file has not header"
    RuntimeError
        "invalid date in the csv file"
    FileNotFoundError
        file name and path is not valid, file doesn't exist.
    """
    # Create output dictionary
    out = {}

    # Check for file
    if isinstance(seriesData, str):
        if delimiter not in seriesData:

            # Open csv file
            fp = open(seriesData, 'r')
        else:
            fp = seriesData.split(delimiter)
            fp = [item for item in fp if len(item) > 0]

        if isinstance(fp, io.TextIOWrapper):

            # Read headers
            line = fp.readline()
        elif isinstance(fp, list):
            line, fp = fp[0], fp[1:]

        if not line:
            logger.error(f"Header in csv file {seriesData} is empty.")
            raise RuntimeError(f"Header in csv file {seriesData} is empty.")
        headers = line.strip().split(',')
        if len(headers) == 0:
            logger.error(f"csv file {seriesData} has no headers.")
            raise RuntimeError(f"csv file {seriesData} has no headers.")

        # Create dictionary, skip first column
        for i, name in enumerate(headers[1:], 1):
            out[name] = Series()

        # Parse rows
        for line in fp:

            # Read items
            items = line.strip().split(',')

            # Add to series
            for i in range(1, len(items)):
                out[headers[i]].add_value(items[0], float(items[i]))

        if isinstance(fp, io.TextIOWrapper):
            fp.close()
    else:
        logger.error(f"File {seriesData} cannot be located.")
        raise FileNotFoundError(f"File {seriesData} cannot be located.")
    return out

def find_start_index(raster, time: Real) -> Integral:
    """find the start index from the RasterFile object.

    Parameters
    ----------
    raster : RasterFile
        an instance of RasterFile object
    time : Real
        time instance value

    Returns
    -------
    Integral
        index of the time step closest to the input time.
    """
    # Initialise left and right indexes
    lIndex = 0
    rIndex = raster.getMaximumTimeIndex()

    # Binary search
    while lIndex+1 < rIndex:

        # Central index
        cIndex = int((lIndex+rIndex)/2)

        # Get central time
        cTime = raster.getTimeFromIndex(cIndex)

        # Update left and right indexes
        if (cTime < time):
            lIndex = cIndex
        elif (cTime > time):
            rIndex = cIndex
        else:
            return cIndex

    return lIndex

def get_series_data(input_series: Union[Dict, Series],
                    xmin: Optional[float] = None,
                    xmax: Optional[float] = None,
                    step_size: Optional[float] = None) -> Dict:
    """Get series data from the series object

    Parameters
    ----------
    input_series : Union[Dict, Series]
        a dictionary with Series object or a Series object
    xmin : Optional[float], optional
        minimum value for x, by default None
    xmax : Optional[float], optional
        maximum value for x, by default None
    step_size : Optional[float], optional
        step size for x, by default None

    Returns
    -------
    Dict
        a dictionary with list of series values (x, y)
    """

    def _get_series_data(inp_data: Series, xmin: Optional[float],
                         xmax: Optional[float], step_size: Optional[float]):
        if xmin is None:
            xmin = inp_data.get_xMin()
        if xmax is None:
            xmax = inp_data.get_xMax()

        if step_size is None:
            nx = 100
            step_size = (xmax - xmin) / nx
        else:
            nx = int(floor(((xmax - xmin) / step_size)))

        out = map(lambda x: [x, inp_data.get(x)], map(lambda x: xmin + x * step_size, range(nx)))
        return list(out)

    dispatcher = partial(_get_series_data, xmin=xmin, xmax=xmax,
                         step_size=step_size)
    if isinstance(input_series, Dict):
        return {item: dispatcher(input_series[item]) for item in input_series}
    elif isinstance(input_series, Series):
        return dispatcher(input_series)

def project_raster(src_raster, src_proj, dst_proj, resolution, **kwargs):

    # Build projections
    if isinstance(src_proj, str):
        _src_proj = ProjectionParameters.from_proj4(src_proj)
    elif isinstance(src_proj, ProjectionParameters):
        _src_proj = src_proj

    if isinstance(dst_proj, str):
        _dst_proj = ProjectionParameters.from_proj4(dst_proj)
    elif isinstance(dst_proj, ProjectionParameters):
        _dst_proj = dst_proj

    if _src_proj != _dst_proj:
        dst_raster = src_raster.reproject(_dst_proj, hx=resolution, hy=resolution)
        return dst_raster
    else:
        return src_raster

class MetRaster:

    def __init__(self, raster, thredds: bool = False, slice: slice = None):

        # Initialise members
        self.raster = raster
        self.currentTimeIndex = 0
        self.rasterMaximumTimeIndex = 0
        self.timeOrigin = 0.0
        self.timeDelta = 0.0
        self.time_A = 0.0
        self.time_B = 0.0

        # Check input raster file
        if isinstance(self.raster, RasterFile):

            # Initialise RasterFile
            self.raster.read(thredds = thredds, layers = slice)

    def setTimeSentinelA(self):
        self.time_A = self.raster.getTime()

    def setTimeSentinelB(self):
        self.time_B = self.raster.getTime()

    def setTimeSentinelAtoB(self):
        self.time_A = self.time_B

    def setProjectionParameters(self, proj_params):
        if isinstance(proj_params, str):
            self.raster.setProjectionParameters(ProjectionParameters.from_proj4(proj_params))
        elif isinstance(proj_params, ProjectionParameters):
            self.raster.setProjectionParameters(proj_params)

    def resetIndex(self, dateTime: datetime):
        if isinstance(self.raster, RasterFile):

            # Update indexes
            time = dateTime.timestamp()
            self.rasterMaximumTimeIndex = self.raster.getMaximumTimeIndex()
            minTime = self.raster.getTimeFromIndex(0)
            maxTime = self.raster.getTimeFromIndex(self.rasterMaximumTimeIndex)
            if time < minTime or time > maxTime:
                minDate = datetime.fromtimestamp(minTime).strftime('%Y-%m-%dT%H:%M:%SZ')
                maxDate = datetime.fromtimestamp(maxTime).strftime('%Y-%m-%dT%H:%M:%SZ')
                err = f'Time {dateTime} out of bounds for met data (range {minDate} to {maxDate})'
                logger.error(err)
                raise RuntimeError(err)

            # Find start time index
            self.currentTimeIndex = find_start_index(self.raster, time)

            # Set indices
            self.raster.setTimeIndex(self.currentTimeIndex)

            # Set time origin and delta
            self.timeOrigin = self.raster.getTime()
            self.timeDelta = time-self.timeOrigin

    def incrementIndex(self):
        if isinstance(self.raster, RasterFile):

            # Increment and check index
            self.currentTimeIndex += 1
            if self.currentTimeIndex > self.rasterMaximumTimeIndex:
                logger.error('Index out of bounds in met file.')
                raise RuntimeError('Index out of bounds in met file.')

            # Set raster index
            self.raster.setTimeIndex(self.currentTimeIndex)

    def decrementIndex(self):
        if isinstance(self.raster, RasterFile):

            # Increment and check index
            self.currentTimeIndex -= 1
            if self.currentTimeIndex < 0:
                logger.error('Index out of bounds in met file.')
                raise RuntimeError('Index out of bounds in met file.')

            # Set raster index
            self.raster.setTimeIndex(self.currentTimeIndex)

    @staticmethod
    def get_raster_handler(metSource):

        # Decide correct handler
        thredds = False
        if 'nc' in pth.splitext(metSource)[-1]:
            backend = 'netcdf'
        elif 'grb' in pth.splitext(metSource)[-1]:
            backend = 'grib'
        elif 'dodsC' in metSource:
            backend, thredds = 'netcdf', True
        else:
            backend = 'gdal'
        check_backend_lib = {"gdal": supported_libs.HAS_GDAL,
                             "netcdf": supported_libs.HAS_NCDF,
                             "grib": supported_libs.HAS_PYGRIB}
        assert check_backend_lib.get(backend, False), f"No library for {backend} is installed."
        return backend, thredds

class SparkModel:
    def __init__(self, cfgJson=None, dtype=np.float32):

        if dtype not in [np.float32, np.float64]:
            logger.error("dtype should be np.float32/np.float64.")
            raise ValueError("dtype should be np.float32/np.float64.")
        else:
            self.dtype = dtype

        if cfgJson is not None:
            self.configFile = deepcopy(cfgJson)
        self.modelInitialised = False
        self.hasMetLayers = False
        self.hasSeries = False
        self.isConfigured = False

        # Create empty defaults
        self.variables = Variables()
        self.sources = Vector()

        self.inputVectors = {}
        self.inputRasters = {}
        self.outputRasters = {}
        self.inputVectorMapping = {}
        self.outputLayerReductions = {}
        self.outputLayerReductionTypes = {}
        self.outputRasterList = []
        self.flatOutputLayers = []

        self.series = None
        self.metLayers = None
        self.gribRasterMap = None
        self.metVariables = {}
        self.seriesVariables = {}

        # Sub models
        self.subModels = {}

        self.firebrandRasterList = []
        self.firebrandCreationScript = ''
        self.firebrandInitialisationScript = ''
        self.firebrandUpdateScript = ''
        self.firebrandAdvectionScript = ''
        self.firebrandTransportScript = ''
        self.firebrandPaths = Vector()
        self.spotFires = Vector()

        # Create empty scripts
        self.advectionScript = ''
        self.buildScript = ''
        self.updateScript = ''
        self.initialisationScript = ''
        self.unique_id = ''

    def configure(self,
                  projection: str = None,
                  resolutionMeters: Union[Integral, Real] = 0,
                  levels: Integral = 1,
                  durationSeconds: Union[Integral, Real] = 0,
                  timeMultiple: Union[Integral, Real] = None,
                  startDateISO8601: str = None):

        if projection is None or projection == '':
            logger.error("Simulation projection must be set.")
            raise ValueError("Simulation projection must be set.")

        if resolutionMeters <= 0:
            logger.error("Resolution must be greater than zero.")
            raise ValueError("Resolution must be greater than zero.")

        if levels <= 0:
            logger.error("Simulation levels must be greater than zero.")
            raise ValueError("Simulation levels must be greater than zero.")

        if durationSeconds <= 0:
            logger.error("Simulation duration must be greater than zero.")
            raise ValueError("Simulation duration must be greater than zero.")

        # Set model configuration
        self.projection = projection
        self.resolutionMeters = resolutionMeters
        self.levels = levels
        self.durationSeconds = durationSeconds
        if timeMultiple is None:
            self.timeMultiple = durationSeconds
        else:
            self.timeMultiple = timeMultiple

        # Get start epoch time
        if startDateISO8601 == '' or startDateISO8601 is None:
            self.startDateISO8601 = ''
            self.startDateTime = 0.0
        else:
            self.startDateISO8601 = startDateISO8601
            self.startDateISO8601 = self.startDateISO8601.replace(' ', '') # Strip all spaces
            self.startDateISO8601 = self.startDateISO8601.replace("Z", "+00:00") # Change Zulu indicator to time offset
            self.startDateTime = datetime.fromisoformat(self.startDateISO8601)
        self.projSim = ProjectionParameters.from_proj4(self.projection)
        self.isConfigured = True
        return self.isConfigured

    def set_input_layers(self, inputLayers: List = None):

        # Read user-defined input layers
        if inputLayers is None or len(inputLayers) == 0:
            return

        for item in inputLayers:

            # Get name
            name = item['name']
            if name is None or name == '':
                logger.error("Unnamed input layer.")
                raise ValueError("Unnamed input layer.")

            if 'source' in item:

                # Get correct handler
                backend, thredds = MetRaster.get_raster_handler(item['source'])

                # Create raster layer
                if 'type' in item:
                    if item['type'] == 'byte':
                        inputType = np.uint8
                    elif item['type'] == 'integer':
                        inputType = np.uint32
                    elif item['type'] == 'float':
                        inputType = np.float32
                    else:
                        err = f"Invalid data type '{item['type']}' specified for input layer {name}."
                        logger.error(err)
                        raise ValueError("" + err)
                else:
                    inputType = np.float32

                self.inputRasters[name] = RasterFile(name = name,
                                         filePath = item['source'],
                                         backend = backend,
                                         data_type = inputType)
                self.inputRasters[name].read(thredds=thredds)

                # Add projection if specified
                if 'projection' in item:
                    self.inputRasters[name].setProjectionParameters(ProjectionParameters.from_proj4(item['projection']))

                # Log layer name
                logger.info(f"  {name}: {item['source']} ({backend} {inputType.__name__})")

            else:

                # Throw error if no source is specified
                err = f"No source specified for input layer {name}."
                logger.error(err)
                raise ValueError("" + err)

    def set_input_vectors(self, inputVectors: List = None):

        # Read user-defined input vectors
        if inputVectors is None or len(inputVectors) == 0:
            return

        for item in inputVectors:

            # Get name
            name = item['name']
            if name is None or name == '':
                logger.error("Unnamed input vector")
                raise ValueError("Unnamed input vector")

            # Get mapping
            mapping = item['mapping']
            if mapping is None or mapping == '':
                err = f"Vector layer {name} must have a mapping type."
                logger.error(err)
                raise ValueError(err)

            # Create empty Vector
            inputVector = Vector()

            # Check type
            if 'source' in item and item['source'] is not None:

                source = item['source']
                if source.endswith('.shp'):

                    # Shapefile source
                    inputVector = shapefileToVector(source)

                elif source.endswith('.geojson') or source.endswith('.json'):

                    # GeoJSON source
                    inputVector = geoJsonToVector(source)

                else:
                    raise ValueError(f"Vector source {source} has unknown type")

                # Project vector
                inputVector = inputVector.convert(self.projSim)

                # Log layer name
                logger.info(f"  {name}: {source} (vector)")

            elif 'data' in item:

                if item['data'] is not None:
                
                    if isinstance(item['data'], (OrderedDict, dict)):

                        # Data is dictionary, assumed to be json
                        inputVector = geoJsonToVector(item['data'], dtype=self.dtype)

                    else:

                        # Data is string, assumed to be json
                        inputVector = geoJsonToVector(json.loads(item['data'],
                                                    object_pairs_hook=OrderedDict),
                                                    dtype=self.dtype)

                # Project vector
                if 'projection' in item:
                    vectorProj = ProjectionParameters.from_proj4(item['projection'])
                    inputVector.setProjectionParameters(vectorProj)
                    inputVector = inputVector.convert(self.projSim)
                else:
                    inputVector.setProjectionParameter(self.projSim)
                    logger.warning(f"Vector layer {name} has no specified projection")

                # Log layer name
                logger.info(f"  {name} (vector)")

            else:

                err = f"Vector layer {name} must contain either 'source' or 'data'."
                logger.error(err)
                raise ValueError("" + err)
                
            # Ensure level property is created
            if not inputVector.hasProperty("level"):        
                for idx in inputVector.getGeometryIndexes():
                    inputVector.setProperty(idx, "level", int(0))
            
            # Expand levels
            if inputVector.hasProperty("levels"):
                for idx in inputVector.getPointIndexes():
                    level_str = inputVector.getProperty(idx, "levels", str)

                    # Parse levels
                    if not level_str:
                        levels = [*range(0, self.levels)]
                    else:
                        levels = []
                        for level in level_str.split(","):
                            try:
                                levels.append(int(level))
                            except ValueError:
                                levels.append(0)

                    # Update point
                    inputVector.setProperty(idx, "level", int(levels[0]))

                    # Copy point
                    for level in levels[1:]:
                        new_idx = inputVector.addPoint(inputVector.getPointCoordinate(idx))
                        inputVector.setProperty(new_idx, "value", inputVector.getProperty(idx, "value", float))
                        inputVector.setProperty(new_idx, "level", int(level))
                
                for idx in inputVector.getLineStringIndexes():
                    level_str = inputVector.getProperty(idx, "levels", str)

                    # Parse levels
                    if not level_str:
                        levels = [*range(0, self.levels)]
                    else:
                        levels = []
                        for level in level_str.split(","):
                            try:
                                levels.append(int(level))
                            except ValueError:
                                levels.append(0)

                    # Update line string
                    inputVector.setProperty(idx, "level", int(levels[0]))

                    # Copy line string
                    for level in levels[1:]:
                        new_idx = inputVector.addLineString(inputVector.getLineStringCoordinates(idx))
                        inputVector.setProperty(new_idx, "value", inputVector.getProperty(idx, "value", float))
                        inputVector.setProperty(new_idx, "level", int(level))
                
                for idx in inputVector.getPolygonIndexes():
                    level_str = inputVector.getProperty(idx, "levels", str)

                    # Parse levels
                    if not level_str:
                        levels = [*range(0, self.levels)]
                    else:
                        levels = []
                        for level in level_str.split(","):
                            try:
                                levels.append(int(level))
                            except ValueError:
                                levels.append(0)

                    # Update polygon
                    inputVector.setProperty(idx, "level", int(levels[0]))

                    # Copy polygon
                    for level in levels[1:]:
                        new_idx = inputVector.addPolygon(inputVector.getPolygonCoordinates(idx))
                        inputVector.setProperty(new_idx, "value", inputVector.getProperty(idx, "value", float))
                        inputVector.setProperty(new_idx, "level", int(level))

                inputVector.removeProperty("levels")   
                
            # Add vector
            self.inputVectors[name] = inputVector 

            # Add mapping
            self.inputVectorMapping[name] = {'type': mapping,
                'script': item.get('script', '') }

            # TODO make rasterise 3D, this is a workaround as rasterise is a 2D operation only
            if self.inputVectorMapping[name]['type'].startswith("rasterise"):
                self.flatOutputLayers.append(name)
            
            # Create output Raster
            outputRaster = Raster(name = name)
            self.outputRasters[name] = outputRaster

    def set_output_layers(self, outputLayers: List = None):

        # Read user-defined output layers
        if outputLayers is None or len(outputLayers) == 0:
            return

        for item in outputLayers:

            # Get name
            name = item['name']
            if name is None or name == '':
                logger.error("Unnamed output layer.")
                raise ValueError("Unnamed output layer.")

            # Store reduction
            if 'reduction' in item:
                self.outputLayerReductionTypes[name] = item['reduction']

            # Create output Raster
            outputRaster = Raster(name = name)

            # Set flat outputs
            if 'flat' in item and item['flat'] == True:
                self.flatOutputLayers.append(name)

            # Set sampling
            outputRaster.setInterpolationType(RasterInterpolationType.Nearest)
            if 'sampling' in item and item['sampling'] == 'linear':
                outputRaster.setInterpolationType(RasterInterpolationType.Bilinear)

            # Update lists
            self.outputRasters[name] = outputRaster

    def set_met_input(self, metLayers: List = None,
                      grib_mapper: Dict = None):

        # Clear variables
        self.metVariables = {}
        self.metLayers = metLayers
        self.metRasters = {} # Map of internal source name to MetRaster
        self.metRasterName = {} # Map of type to internal source name
        self.gribRasterMap = {} # Grib map
        advectionScriptHead = ''

        # Create conversion and offset map
        self.metConversion = {}
        self.metOffset = {}

        # Check layers
        if metLayers is None or len(metLayers) == 0:
            return

        # Check for start date
        if self.startDateISO8601 == '':
            logger.warn('Start date must be specified if using met data')
            warnings.warn('Start date must be specified if using met data', RuntimeWarning)

        # Create mapping scripts
        self.scriptMetMapA = ''
        self.scriptMetMapB = ''
        self.scriptMetMapAB = ''

        # Parse layers
        for item in metLayers:

            # Check data
            if 'source' not in item:
                logger.error('Met layers must have a specified source.')
                raise RuntimeError('Met layers must have a specified source.')
            metSource = item['source']

            # Get name
            if 'name' not in item or item['name'] == '':
                logger.error('Met layers must be named')
                raise RuntimeError('Met layers must be named.')
            metName = item['name']

            # Get type
            if 'type' not in item:
                logger.error('Met layers must have a specified type')
                raise RuntimeError('Met layers must have a specified type.')
            metType = item['type']

            # Check for existing entry
            if metType in self.seriesVariables:
                continue
            self.metRasterName[metType] = metName

            # Update conversion and offset
            if 'conversion' in item:
                self.metConversion[metType] = item['conversion']
            if 'offset' in item:
                self.metOffset[metType] = item['offset']

            # Get correct handler
            backend, thredds = MetRaster.get_raster_handler(metSource)

            # Get grib map if specified when reading grib files
            self.gribRasterMap[metName] = None
            if backend == "grib":
                if grib_mapper is None:
                    raise ValueError("grib map should be specified")
                if metType not in grib_mapper:
                    raise ValueError("grib map must contain type {metType}")
                self.gribRasterMap[metName] = {metName: grib_mapper[type]}

            # Create raster layer
            self.metRasters[metName] = MetRaster(RasterFile(name=metName,
                                                          filePath=metSource,
                                                          variable_map=self.gribRasterMap[metName],
                                                          backend=backend,), thredds)

            if 'projection' in item:
                self.metRasters[metName].setProjectionParameters(item['projection'])

            # Set interpolation
            if metType == 'wind_direction':
                self.metRasters[metName].raster.setInterpolationType(RasterInterpolationType.Nearest)
            else:
                self.metRasters[metName].raster.setInterpolationType(RasterInterpolationType.Bilinear)

            # Capture special variable names
            if metType == 'wind_magnitude':
                self.metVariables[metType] = 'wind_speed'
            elif metType == 'temperature':
                self.metVariables[metType] = 'temp'
            elif metType == 'dew_point_temperature':
                self.metVariables[metType] = 'dew_temp'
            elif metType == 'relative_humidity':
                self.metVariables[metType] = 'rel_hum'
            else:
                self.metVariables[metType] = metType
            raster_var = self.metVariables[metType]

            # Create variables
            self.variables.set(f"time_origin_{raster_var}", 0.0)
            self.variables.set(f"time_length_{raster_var}", 0.0)

            # Create rasters
            if not metType.startswith('wind'):

                # Add internal rasters
                self.metRasters[raster_var] = MetRaster(Raster(name = raster_var))
                self.outputRasters[raster_var] = self.metRasters[raster_var].raster

                # Update advection script
                advectionScriptHead += f'''
                    REAL delta_{raster_var} = (time-time_origin_{raster_var})/time_length_{raster_var};
                    {raster_var} = (1.0-delta_{raster_var})*{raster_var}_A+delta_{raster_var}*{raster_var}_B;
                '''

            # Add internal interpolation rasters
            self.metRasters[f'{raster_var}_A'] = MetRaster(Raster(name = f'{raster_var}_A'))
            self.metRasters[f'{raster_var}_A'] = MetRaster(Raster(name = f'{raster_var}_A'))
            self.outputRasters[f'{raster_var}_A'] = self.metRasters[f'{raster_var}_A'].raster
            self.metRasters[f'{raster_var}_B'] = MetRaster(Raster(name = f'{raster_var}_B'))
            self.outputRasters[f'{raster_var}_B'] = self.metRasters[f'{raster_var}_B'].raster

            # Create script for current time step
            self.scriptMetMapA += f'''
            // Map {metType}
            {raster_var}_A = %s*%f+%f;
            ''' % (self.metRasterName[metType],
                   self.metConversion.get(metType, 1),
                   self.metOffset.get(metType, 0))

            # Create script for next time step
            self.scriptMetMapB += f'''
            // Map {metType}
            {raster_var}_B = %s*%f+%f;
            ''' % (self.metRasterName[metType],
                   self.metConversion.get(metType, 1),
                   self.metOffset.get(metType, 0))

            # create script for swapping raster time instances
            self.scriptMetMapAB += f'''
            // Update {raster_var}
            {raster_var}_A = {raster_var}_B;
            {raster_var}_B = %s*%f+%f;
            ''' % (self.metRasterName[metType],
                   self.metConversion.get(metType, 1),
                   self.metOffset.get(metType, 0))

            # Log layer name
            logger.info(f"  {metName}: {metSource}, {metType} ({backend})")
            if metType in self.metConversion:
                logger.info(f"    scale: {self.metConversion[metType]} ({backend})")
            if metType in self.metOffset:
                logger.info(f"    offset: {self.metOffset[metType]} ({backend})")

        # Set advection script to interpolate met rasters
        if ('wind_direction' in self.metRasterName and 'wind_magnitude' in self.metRasterName):
            advectionScriptHead += '''
                REAL delta_wind = (time-time_origin_wind_speed)/time_length_wind_speed;

                REAL wind_x_A = -wind_speed_A*sin(radians(wind_direction_A));
                REAL wind_y_A = -wind_speed_A*cos(radians(wind_direction_A));

                REAL wind_x_B = -wind_speed_B*sin(radians(wind_direction_B));
                REAL wind_y_B = -wind_speed_B*cos(radians(wind_direction_B));

                advect_x = (1.0-delta_wind)*wind_x_A+delta_wind*wind_x_B;
                advect_y = (1.0-delta_wind)*wind_y_A+delta_wind*wind_y_B;
            '''
        elif ('wind_x' in self.metRasterName and 'wind_y' in self.metRasterName):
            advectionScriptHead += '''
                REAL delta_wind_x = (time-time_origin_wind_x)/time_length_wind_x;
                REAL delta_wind_y = (time-time_origin_wind_y)/time_length_wind_y;

                advect_x = (1.0-delta_wind_x)*wind_x_A+delta_wind_x*wind_x_B;
                advect_y = (1.0-delta_wind_y)*wind_y_A+delta_wind_y*wind_y_B;
            '''

        # Update advection script
        self.advectionScript = advectionScriptHead + self.advectionScript
        
        # Set flag for met layer data
        self.hasMetLayers = True

    def set_series_input(self, seriesCSVData: str = None):

        # Clear variables
        self.seriesVariables = {}
        advectionScriptHead = ''

        # Check series
        self.series = {}
        if seriesCSVData is None or seriesCSVData == "":
            return
        self.hasSeries = True

        # Check for start date
        if self.startDateISO8601 == '':
            logger.error("Start date must be specified if using met data.")
            raise RuntimeError("Start date must be specified if using met data.")

        # Parse series
        self.series = parse_series_file(seriesCSVData)

        # Change series names to special variable names
        if 'wind_magnitude' in self.series:
            self.series['wind_speed'] = self.series.pop('wind_magnitude')
        if 'temperature' in self.series:
            self.series['temp'] = self.series.pop('temperature')
        if 'dew_point_temperature' in self.series:
            self.series['dew_temp'] = self.series.pop('dew_point_temperature')
        if 'relative_humidity' in self.series:
            self.series['rel_hum'] = self.series.pop('relative_humidity')

        # Set wind series parameters
        if 'wind_speed' in self.series:
            self.series['wind_speed'].setInterpolation(SeriesInterpolationType.MonotoneCubic)

        if 'wind_direction' in self.series:
            self.series['wind_direction'].setInterpolation(SeriesInterpolationType.BoundedLinear)
            self.series['wind_direction'].setBounds(0.0, 360.0)

        # Create variables table
        for name in self.series:
            if name == 'wind_speed':
                self.seriesVariables['wind_magnitude'] = 'wind_speed'
            elif name == 'temp':
                self.seriesVariables['temperature'] = 'temp'
            elif name == 'dew_temp':
                self.seriesVariables['dew_point_temperature'] = 'dew_temp'
            elif name == 'rel_hum':
                self.seriesVariables['relative_humidity'] = 'rel_hum'
            else:
                self.seriesVariables[name] = name

        # Create variables
        for metVar in self.seriesVariables.values():
            self.variables.set(metVar, 0.0)

        # Set advection script
        if 'wind_magnitude' in self.seriesVariables and 'wind_direction' in self.seriesVariables:
            advectionScriptHead += '''
                advect_x = -wind_speed*sin(radians(wind_direction));
                advect_y = -wind_speed*cos(radians(wind_direction));
                '''
        elif ('wind_x' in self.seriesVariables and 'wind_y' in self.seriesVariables):
            advectionScriptHead += '''
                advect_x = wind_x;
                advect_y = wind_y;
            '''
            
        # Update advection script
        self.advectionScript = advectionScriptHead + self.advectionScript

    def set_variable(self, name: Optional[str] = None,
                     value: Optional[Union[Real, Iterable[Real]]] = None,
                     variablesCSVData : Optional[Union[str, List[str]]] = None):
        """set a variable

        Parameters
        ----------
        name : str, optional
            name of a variable, by default None
        value : Union[Real, Iterable[Real]], optional
            value of the variable, by default None
        variablesCSVData : Optional[Union[str, List[str]]]
            path and name of the CSV file, or string of CSV data.
        """
        if all(map(lambda s: s is None, [name, value, variablesCSVData])):
            return

        if name is not None and value is not None:
            self.variables.set(name, value)

        if variablesCSVData is not None:
            if isinstance(variablesCSVData, str):
                # handle case where this is a CSV str or file
                # Parse variables from csv
                variable_array_values = parse_variables_file(variablesCSVData)

                # set variables parsed from CSV
                if variable_array_values:
                    for key in variable_array_values:
                        self.variables.set(key, variable_array_values[key])
            elif isinstance(variablesCSVData, list):
                # handle case where this is list of CSV str or files
                for item in variablesCSVData:
                    # Parse variables from csv
                    variable_array_values = parse_variables_file(item)

                    # set variables parsed from CSV
                    if variable_array_values:
                        for key in variable_array_values:
                            self.variables.set(key, variable_array_values[key])



    def get_variable(self, name: Optional[str] = None) -> Dict:
        """get a variable.

        Parameters
        ----------
        name : Optional[str], optional
            name of the Variable, by default None

        Returns
        -------
        Dict
            a dictionary with name of variable and its value
        """

        out = {}
        if self.variables.hasData():
            var_idx = self.variables.getIndexes()
            if name is None:
                for item in var_idx:
                    out[item] = self.variables.get(item)
                    if isinstance(out[item], np.ndarray):
                        out[item] = out[item].tolist()
            else:
                if name in var_idx:
                    out[name] = self.variables.get(name)
                    if isinstance(out[name], np.ndarray):
                        out[name] = out[name].tolist()
        return out

    def set_advection_model(self, advectionModel: str = None):
        self.advectionScript = advectionModel
        
        # Patch advection script
        self.advectionScript = re.sub('\\blevel\\b', 'kpos',
                                  self.advectionScript)

    def set_initialisation_model(self, initialisationModel: str = None):
        self.initialisationScript = initialisationModel
        
        # Patch initialisation script
        self.initialisationScript = re.sub('\\blevel\\b', 'kpos',
                                  self.initialisationScript)

    def set_ros_models(self, rosModels: Union[str, Dict] = None):
        self.buildScript = ""
        if rosModels is None or rosModels == "":
            return self.buildScript
        if isinstance(rosModels, str):
            _rosModels = json.loads(rosModels,
                                       object_pairs_hook=OrderedDict)
            self.buildScript = parse_multiple_models(_rosModels,
                                                  "rateOfSpread",
                                                  "speed = 0.0;")
        else:
            self.buildScript = parse_multiple_models(rosModels,
                                                   "rateOfSpread",
                                                   "speed = 0.0;")

        # Patch build script
        self.buildScript = re.sub('\\blevel\\b', 'kpos',
                                  self.buildScript)
        self.buildScript = re.sub('\\bwind\\b', 'advect_dot_normal',
                                  self.buildScript)
        self.buildScript = re.sub('\\bwind_vector\\b', 'advect_vector',
                                  self.buildScript)
        self.buildScript = re.sub('\\wind_speed\\b',
                                  'advect_mag',
                                  self.buildScript)
        return self.buildScript

    def set_update_models(self, updateModels: Union[str, Dict] = None):

        self.updateScript = ""
        if updateModels is None or updateModels == "":
            return self.updateScript
        if isinstance(updateModels, str):
            _updateModels = json.loads(updateModels,
                                        object_pairs_hook=OrderedDict)
            self.updateScript = parse_multiple_models(_updateModels,
                                                    "update")
        else:
            self.updateScript = parse_multiple_models(updateModels,
                                                    "update")

        # Patch update script
        self.updateScript = re.sub('\\blevel\\b', 'kpos',
                                  self.updateScript)
        self.updateScript = re.sub('\\bwind\\b',
                                   'advect_dot_normal',
                                   self.updateScript)
        self.updateScript = re.sub('\\bwind_vector\\b',
                                   'advect_vector',
                                   self.updateScript)
        self.updateScript = re.sub('\\wind_speed\\b',
                                  'advect_mag',
                                  self.updateScript)
        return self.updateScript

    def set_sources(self, sources: Union[Dict, str] = None):

        # Ensure sources is a dict
        if isinstance(sources, str):
            sources = json.loads(sources, object_pairs_hook=OrderedDict)

        # Convert to Vector
        self.sources = geoJsonToVector(sources, dtype=self.dtype)
        if not self.sources.hasData():
            logger.error("Source GeoJSON contains no data.")
            raise RuntimeError("Source GeoJSON contains no data.")

        # Ensure radius is 3x resolution or greater
        if self.sources.hasProperty("radius"):
            for idx in chain(self.sources.getPointIndexes(), self.sources.getLineStringIndexes()):
                radius = self.sources.getProperty(idx, "radius", float)
                if radius < 3.0*self.resolutionMeters:
                    self.sources.setProperty(idx, "radius", 3.0*self.resolutionMeters)
        else:
            for idx in chain(self.sources.getPointIndexes(), self.sources.getLineStringIndexes()):
                self.sources.setProperty(idx, "radius", 3.0*self.resolutionMeters)
                
        # Ensure level property is created
        if not self.sources.hasProperty("level"):        
            for idx in self.sources.getGeometryIndexes():
                self.sources.setProperty(idx, "level", int(0))
        
        # Expand levels
        if self.sources.hasProperty("levels"):
            for idx in self.sources.getPointIndexes():
                level_str = self.sources.getProperty(idx, "levels", str)

                # Parse levels
                if not level_str:
                    levels = [*range(0, self.levels)]
                else:
                    levels = []
                    for level in level_str.split(","):
                        try:
                            levels.append(int(level))
                        except ValueError:
                            levels.append(0)

                # Update point
                self.sources.setProperty(idx, "level", int(levels[0]))

                # Copy Point
                for level in levels[1:]:
                    new_idx = self.sources.addPoint(self.sources.getPointCoordinate(idx))
                    self.sources.setProperty(new_idx, "radius", self.sources.getProperty(idx, "radius", float))
                    self.sources.setProperty(new_idx, "level", int(level))
                    
            for idx in self.sources.getLineStringIndexes():
                level_str = self.sources.getProperty(idx, "levels", str)

                # Parse levels
                if not level_str:
                    levels = [*range(0, self.levels)]
                else:
                    levels = []
                    for level in level_str.split(","):
                        try:
                            levels.append(int(level))
                        except ValueError:
                            levels.append(0)

                # Update line string
                self.sources.setProperty(idx, "level", int(levels[0]))

                # Copy line string
                for level in levels[1:]:
                    new_idx = self.sources.addLineString(self.sources.getLineStringCoordinates(idx))
                    self.sources.setProperty(new_idx, "radius", self.sources.getProperty(idx, "radius", float))
                    self.sources.setProperty(new_idx, "level", int(level))
                    
            for idx in self.sources.getPolygonIndexes():
                level_str = self.sources.getProperty(idx, "levels", str)

                # Parse levels
                if not level_str:
                    levels = [*range(0, self.levels)]
                else:
                    levels = []
                    for level in level_str.split(","):
                        try:
                            levels.append(int(level))
                        except ValueError:
                            levels.append(0)

                # Update polygon
                self.sources.setProperty(idx, "level", int(levels[0]))

                # Copy polygon
                for level in levels[1:]:
                    new_idx = self.sources.addPolygon(self.sources.getPolygonCoordinates(idx))
                    self.sources.setProperty(new_idx, "level", int(level))

            self.sources.removeProperty("levels")           
        
                
        # Set origin point of simulation, used to normalise positions in the particle model
        bounds = self.sources.getBounds()
        bounds = bounds.convert(self.projSim, self.sources.getProjectionParameters())
        self.origin = bounds.centroid

    def set_submodels(self, subModels: Union[str, Dict] = None):

        if subModels is None:
            return
        if isinstance(subModels, str):
            _subModels = json.loads(subModels)
        else:
            _subModels = subModels

        # Search for specific models
        if 'firebrands' in _subModels:
            if 'enable' in _subModels['firebrands'] and _subModels['firebrands']['enable']:
                logger.info("Using firebrand sub-model")
                self.subModels['firebrands'] = _subModels['firebrands']

                # Create particles
                firebrands = Vector()
                firebrands.setProjectionParameters(self.projSim)

                # Create spot fires
                self.spotFires = Vector()
                self.spotFires.setProjectionParameters(self.projSim)
                self.firebrandPaths = Vector()
                self.firebrandPaths.setProjectionParameters(self.projSim)

                # Create particle solver
                self.firebrandSolver = Particle()

                if 'firebrandCreationModel' in self.subModels['firebrands']:
                    self.firebrandCreationScript = self.subModels['firebrands']['firebrandCreationModel']
                    self.firebrandCreationScript = re.sub('\\bclass\\b', '((classbits&0xFFFFFE)>>1)',
                        self.firebrandCreationScript)
                    self.firebrandCreationScript = re.sub('\\bsubclass\\b', '(classbits>>24)',
                        self.firebrandCreationScript)
                    self.firebrandCreationScript = '''

                        // Set defaults
                        level = kpos;

                        // Store origin
                        origin_x = x;
                        origin_y = y;

                        // Set default velocity
                        REALVEC3 velocity = (REALVEC3)(0.0, 0.0, 0.0);

                        // Set advection from gridded values
                        REALVEC3 advect = (REALVEC3)(advect_x, advect_y, 0.0);
                    ''' + self.firebrandCreationScript + '''

                        // Store velocity
                        firebrand_vx = velocity.x;
                        firebrand_vy = velocity.y;
                        firebrand_vz = velocity.z;

                        // Store initial advection
                        firebrand_advect_x = advect.x;
                        firebrand_advect_y = advect.y;
                        firebrand_advect_z = advect.z;
                    '''

                if 'firebrandInitialisationModel' in self.subModels['firebrands']:
                    self.firebrandInitialisationScript = self.subModels['firebrands']['firebrandInitialisationModel']
                    self.firebrandInitialisationScript = '''

                        // Default radius
                        radius = 1.0;

                        // Default velocity
                        velocity.x = firebrand_vx;
                        velocity.y = firebrand_vy;
                        velocity.z = firebrand_vz;
                    ''' + self.firebrandInitialisationScript

                if 'firebrandAdvectionModel' in self.subModels['firebrands']:
                    self.firebrandAdvectionScript = self.subModels['firebrands']['firebrandAdvectionModel']
                    self.firebrandAdvectionScript = '''

                        // Set to initial advection
                        advect.x = firebrand_advect_x;
                        advect.y = firebrand_advect_y;
                        advect.z = firebrand_advect_z;
                    ''' + self.firebrandAdvectionScript

                if 'firebrandUpdateModel' in self.subModels['firebrands']:
                    self.firebrandUpdateScript = self.subModels['firebrands']['firebrandUpdateModel']
                self.firebrandUpdateScript += '''
                    if (sample_plane_cross) {{

                        // Set negative radius to turn off particle processing
                        radius = -radius;

                        // Correct particle to impact location
                        position = position_sample_plane;

                    }} else {{

                        // Update particle clock
                        time += dt;
                    }}
                    '''

                if 'firebrandTransportModel' in self.subModels['firebrands']:
                    self.firebrandTransportScript = self.subModels['firebrands']['firebrandTransportModel']

                firebrands.addProperty("level")
                firebrands.addProperty("origin_x")
                firebrands.addProperty("origin_y")
                firebrands.addProperty("firebrand_vx")
                firebrands.addProperty("firebrand_vy")
                firebrands.addProperty("firebrand_vz")
                firebrands.addProperty("firebrand_advect_x")
                firebrands.addProperty("firebrand_advect_y")
                firebrands.addProperty("firebrand_advect_z")
                config = {
                    "dt": 1.0,
                    "initialisationScript" : self.firebrandInitialisationScript,
                    "advectionScript" : self.firebrandAdvectionScript,
                    "postUpdateScript" : self.firebrandUpdateScript,
                    "updateScript" : self.firebrandTransportScript,
                    "samplingPlane": { "normal": [0, 0, 1], "point": [0, 0, 0] }
                }
                self.firebrandSolver.init(json.dumps(config), firebrands)

    def initialise_solver(self):

        # Create Spark solver configuration
        self.solverConfig = {
            "startDateISO8601": self.startDateISO8601,
            "projection": self.projection,
            "resolution": self.resolutionMeters,
            "levels": self.levels,
            "timeMultiple": self.timeMultiple,
            "advectionScript": self.advectionScript,
            "initialisationScript": self.initialisationScript,
            "buildScript": self.buildScript,
            "updateScript": self.updateScript,
            "flatOutputLayers": self.flatOutputLayers
        }

        # Build lists
        inputList = RasterPtrList()
        for raster in self.inputRasters.values():
            inputList.append(raster)
        outputList = RasterPtrList()
        for raster in self.outputRasters.values():
            outputList.append(raster)

        # Initialise solver
        self.solver = LevelSet()
        rc = self.solver.init(json.dumps(self.solverConfig),
                              self.sources,
                              self.variables,
                              inputList,
                              outputList)
        if not rc:
            logger.error("Unable to initialise solver.")
            raise RuntimeError("Unable to initialise solver.")

        # Store simulation dimensions
        self.simulationsDims = RasterDimensions()

        # Initialise met layers
        if self.hasMetLayers and self.metRasters:

            # Reset indexes
            for metRaster in self.metRasters.values():
                metRaster.resetIndex(self.startDateTime)

            # Map met layer A
            scriptRasters = []
            for metType, metVar in self.metVariables.items():
                metRaster = self.metRasters[self.metRasterName[metType]]
                metRaster.setTimeSentinelA()
                self.variables.set(f"time_origin_{metVar}", metRaster.time_A-metRaster.timeOrigin-metRaster.timeDelta)
                scriptRasters += [
                    self.metRasters[f"{metVar}_A"].raster,
                    metRaster.raster]
            runScript(self.scriptMetMapA, scriptRasters)

            # Increment indexes
            for metType, metVar in self.metVariables.items():
                self.metRasters[self.metRasterName[metType]].incrementIndex()

            # Map met layer B
            scriptRasters = []
            for metType, metVar in self.metVariables.items():
                metRaster = self.metRasters[self.metRasterName[metType]]
                metRaster.setTimeSentinelB()
                self.variables.set(f"time_length_{metVar}", metRaster.time_B-metRaster.time_A)
                scriptRasters += [
                    self.metRasters[f"{metVar}_B"].raster,
                    metRaster.raster]
            runScript(self.scriptMetMapB, scriptRasters)

        # Save start epoch time
        self.startEpochMilliseconds = self.solver.getEpochMilliseconds()
        self.lastEpochMilliseconds = self.startEpochMilliseconds

        # Update data
        self.update_input_data(True)

        # Take first step
        self.solver.step()

        # Reset report time
        self.nextReportTime = 0.0

        # Build output list, ensuring flat layers are last
        self.outputRasterList = []
        for name in self.outputRasters:
            if name not in self.flatOutputLayers:
                self.outputRasterList.append(self.outputRasters[name])
        for name in self.flatOutputLayers:
            self.outputRasterList.append(self.outputRasters[name])

        self.modelInitialised = True

    def update_input_data(self, init: bool = False):

        # Check for dimension changes
        changedDimensions = False
        currentDims = self.solver.getClassification().getDimensions()
        if (self.simulationsDims.tx, self.simulationsDims.ty) != (currentDims.tx, currentDims.ty):

            # Rebuild raster lists # TODO this is a workaround as getAdvect is a deep rather than shallow copy
            self.firebrandRasterList = []
            for name in self.outputRasters:
                self.firebrandRasterList.append(self.outputRasters[name])
            self.firebrandRasterList.append(self.solver.getAdvect_x())
            self.firebrandRasterList.append(self.solver.getAdvect_y())
            self.firebrandRasterList.append(self.solver.getClassification())

            # Update dimensions
            changedDimensions = True
            self.simulationsDims = currentDims

        # Get epoch time
        currentEpochMilliSeconds = self.solver.getEpochMilliseconds()

        # Update mapping
        if self.inputVectors:

            # Create timeslice
            if changedDimensions:
                lowerTimeBound = 0
            else:
                lowerTimeBound = (self.lastEpochMilliseconds-self.startEpochMilliseconds)/1000
            upperTimeBound = (currentEpochMilliSeconds-self.startEpochMilliseconds)/1000
            if init or (upperTimeBound-lowerTimeBound) > 0:

                # Create time-sliced bounding box
                b = BoundingBox()
                c_min = b[1]
                c_max = b[0]
                c_min.s = lowerTimeBound
                c_max.s = upperTimeBound
                b = BoundingBox.from_list([c_min.to_list(), c_max.to_list()])

                # Update vector layers
                for name in self.inputVectors:

                    # Get timesliced vector
                    inputVector = self.inputVectors[name].region(b)

                    if inputVector.hasData():

                        # Apply distance mapping
                        if self.inputVectorMapping[name]['type'].startswith("distance"):

                            # Clear raster
                            self.outputRasters[name].setAllCellValues(self.outputRasters[name].nullValue)

                            # Map vector
                            self.outputRasters[name].mapVector(inputVector,
                                self.inputVectorMapping[name]['script'], levelPropertyName = 'level')

                        # Apply rasterisation mapping
                        elif self.inputVectorMapping[name]['type'].startswith("rasterise"):
                            self.outputRasters[name].rasterise(inputVector,
                                self.inputVectorMapping[name]['script'])


        # Update series data
        if self.hasSeries:
            for name in self.series:
                self.variables.set(name,
                    self.series[name].get(currentEpochMilliSeconds))

        # Update met data
        if self.hasMetLayers and self.metRasters:

            # Update met times
            needsUpdate = False
            for metType, metVar in self.metVariables.items():

                # Get raster
                metRaster = self.metRasters[self.metRasterName[metType]]
                if currentEpochMilliSeconds/1000 > metRaster.time_B:

                    # Increment indexes
                    metRaster.incrementIndex()

                    # Update met grid times
                    metRaster.setTimeSentinelAtoB()
                    metRaster.setTimeSentinelB()

                    # Update met variables
                    self.variables.set(f"time_origin_{metVar}", metRaster.time_A-metRaster.timeOrigin-metRaster.timeDelta)
                    self.variables.set(f"time_length_{metVar}", metRaster.time_B-metRaster.time_A)

                    needsUpdate = True

            if needsUpdate:

                # Update met grid (swap)
                scriptRasters = []
                for metType, metVar in self.metVariables.items():
                    scriptRasters += [
                        self.metRasters[f"{metVar}_A"].raster,
                        self.metRasters[f"{metVar}_B"].raster,
                        self.metRasters[self.metRasterName[metType]].raster]
                runScript(self.scriptMetMapAB, scriptRasters)

            elif changedDimensions:

                # Decrement indexes
                for metType, metVar in self.metVariables.items():
                    self.metRasters[self.metRasterName[metType]].decrementIndex()

                # Map met layer A
                scriptRasters = []
                for metType, metVar in self.metVariables.items():
                    scriptRasters += [
                        self.metRasters[f"{metVar}_A"].raster,
                        self.metRasters[self.metRasterName[metType]].raster]
                runScript(self.scriptMetMapA, scriptRasters)

                # Increment indexes
                for metType, metVar in self.metVariables.items():
                    self.metRasters[self.metRasterName[metType]].incrementIndex()

                # Map met layer B
                scriptRasters = []
                for metType, metVar in self.metVariables.items():
                    scriptRasters += [
                        self.metRasters[f"{metVar}_B"].raster,
                        self.metRasters[self.metRasterName[metType]].raster]
                runScript(self.scriptMetMapB, scriptRasters)

    def run_model(self, duration: Union[Integral, Real] = 0.0):

        # Check for initialisation
        if not self.modelInitialised:
            logger.error("Spark model is not initialized.")
            raise RuntimeError("Spark model is not initialized.")

        # Update report time
        self.nextReportTime += duration

        # Set return to true if solver time is already greater than report time
        rc = self.solver_time >= self.nextReportTime

        while self.solver_time < min(self.durationSeconds, self.nextReportTime):

            # Update data
            self.update_input_data()

            # Run sub-models
            for subModel in self.subModels:

                # Firebrand model
                if subModel == 'firebrands':

                    # Firebrand creation
                    # To increase numerical precision, a simulation origin is subtracted from the particle positions
                    if self.firebrandCreationScript:
                        newFireBrands = stipple(self.firebrandCreationScript + f'''
                            x -= {self.origin[0]};
                            y -= {self.origin[1]};
                            ''',
                            self.firebrandRasterList, [
                                "level", "origin_x", "origin_y",
                                "firebrand_vx", "firebrand_vy", "firebrand_vz",
                                "firebrand_advect_x", "firebrand_advect_y", "firebrand_advect_z"])
                        for idx in newFireBrands.getPointIndexes():
                            newFireBrands.setProperty(idx, "start_time", self.solver_time)
                        self.firebrandSolver.addParticles(newFireBrands)

                    # Create new firebrands vectors
                    newSpotFires = Vector()
                    newSpotFires.setProjectionParameters(self.projSim)
                    newFirebrandPaths = Vector()
                    newFirebrandPaths.setProjectionParameters(self.projSim)

                    # Run firebrand model
                    self.firebrandSolver.setTimeStep(self.solver.parameters.dt)
                    self.firebrandSolver.step()

                    # Create spot fires from firebrands
                    if (self.firebrandSolver.getSamplePlaneIndexCount() > 0):

                        # Populate new spot fires from sample plane particles
                        firebrandParticles = self.firebrandSolver.getParticles()
                        for idx in self.firebrandSolver.getSamplePlaneIndexes():

                            # Get sample plane particle coordinate
                            c = firebrandParticles.getPointCoordinate(idx).to_list()

                            # Check firebrand time
                            if c[3] > 0.0:

                                # Translate to simulation origin
                                c[0] += self.origin[0]
                                c[1] += self.origin[1]

                                # Set ignition time
                                c[3] = self.solver_time

                                # Add to new spot fires
                                fid = newSpotFires.addPoint(c)
                                newSpotFires.setProperty(fid, "radius", float(3.0*self.resolutionMeters))
                                newSpotFires.setProperty(fid, "time", self.solver_time)
                                newSpotFires.setProperty(fid, "level", firebrandParticles.getProperty(idx, "level", int))

                                # Record path
                                pid = newFirebrandPaths.addLineString([[
                                    firebrandParticles.getProperty(idx, "origin_x", float),
                                    firebrandParticles.getProperty(idx, "origin_y", float)], [c[0], c[1]]])
                                newFirebrandPaths.setProperty(pid, "time", firebrandParticles.getProperty(idx, "start_time", float))
                                newFirebrandPaths.setProperty(pid, "end_time", self.solver_time)
                                newFirebrandPaths.setProperty(pid, "level", firebrandParticles.getProperty(idx, "level", int))

                    # Add new firebrands
                    if newSpotFires.hasData():
                        self.solver.addSource(newSpotFires)
                        self.spotFires += newSpotFires
                    if newFirebrandPaths.hasData():
                        self.firebrandPaths += newFirebrandPaths

            # Step
            self.lastEpochMilliseconds = self.solver.getEpochMilliseconds()
            rc = self.solver.step()
            if not rc:
                break

        return rc

    def resize_boundary(self, n: int):
        if not self.modelInitialised:
            raise RuntimeError("Model is not yet initialised.")

        if n <= 0:
            raise RuntimeError("Boundary resize parameter must be positive.")

        # Get classification
        currentDims = self.solver.getClassification().getDimensions()

        # Add n tiles to boundary
        self.solver.resizeDomain(currentDims.tx+2*n, currentDims.ty+2*n, n, n)

        # Update data
        self.update_input_data()

        return None

    def compare_output(self, inpFile: str = None,
                       outFile: str=None, method='jaccard'):
        score = None

        # Get simulation extent
        simulationExtent = self.solver.getArrival()
        simulationExtent.name = 'simulationExtent'

        # Get recorded extent
        recordedVector = geoJsonToVector(inpFile)
        if not recordedVector.hasData():
            logger.error("Invalid comparison GeoJSON.")
            raise RuntimeError("Invalid comparison GeoJSON.")
        recordedVector = recordedVector.convert(self.projSim)
        recordedExtent = recordedVector.rasterise(self.resolutionMeters,
                                                  "output = 1.0;",
                                                  GeometryType.Polygon,
                                                  simulationExtent.getBounds())
        recordedExtent.name = 'recordedExtent'

        # Calculate comparison
        if method == "jaccard":
            score = jaccard_score(simulationExtent, recordedExtent, outFile=outFile)

        return score

    def write_output_rasters(self, outputLayers: List = None,
                             outputProjection: str = None):
        if len(self.outputRasters) == 0:
            return

        # Write raster layers to tiff file
        if outputProjection is None:
            projOut = self.projSim
        else:
            projOut = ProjectionParameters.from_proj4(outputProjection)


        for item in outputLayers:
            if 'destination' in item:

                # Get Raster
                out_name = item['name']
                outputRaster = self.outputRasters[out_name]

                # Reproject raster
                outputRaster = project_raster(outputRaster,
                                              self.projSim,
                                              projOut,
                                              self.resolutionMeters)
                if outputRaster.name != out_name:
                    outputRaster.name = out_name

                # Write raster
                out_file = item['destination']
                out_description = item['description']
                outputRaster.write(out_file, json.dumps({"description": out_description}))

    def process_output(self, processingScript: str = None):

        if processingScript is None:
            return

        # Build processing list
        processingRasterList = []
        if processingScript is not None:
            for name in self.flatOutputLayers:
                processingRasterList.append(self.outputRasters[name])
            for name in self.outputRasters:
                if name not in self.flatOutputLayers:
                    processingRasterList.append(self.outputRasters[name])

        if len(processingRasterList) > 0:

            # Set reduction
            for name in self.outputRasters:
                if name in self.outputLayerReductionTypes:
                    outputRaster = self.outputRasters[name]
                    reduction = self.outputLayerReductionTypes[name]
                    if reduction is not None:
                        reduction = reduction.lower()
                        if reduction == 'maximum':
                            outputRaster.setReductionType(ReductionType.Maximum)
                        elif reduction == 'minimum':
                            outputRaster.setReductionType(ReductionType.Minimum)
                        elif reduction == 'sum':
                            outputRaster.setReductionType(ReductionType.Sum)
                        elif reduction == 'count':
                            outputRaster.setReductionType(ReductionType.Count)
                        elif reduction == 'mean':
                            outputRaster.setReductionType(ReductionType.Mean)

            # Run post-processing
            runScript(processingScript, processingRasterList)

            # Store output reductions
            self.outputLayerReductions = {}
            for name in self.outputLayerReductionTypes:
                reduction = self.outputLayerReductionTypes[name]
                if reduction is not None:
                    reduction = reduction.lower()
                    if reduction != 'none':

                        # Get value
                        val = self.outputRasters[name].reduceVal

                        # Set null values
                        if np.isnan(val):
                            val = None

                        # Set dictionary
                        self.outputLayerReductions[name] = {
                            "type": reduction,
                            "value": val
                        }

            # Reset reductions
            for name in self.outputRasters:
                if name in self.outputLayerReductionTypes:
                    outputRaster = self.outputRasters[name]
                    outputRaster.setReductionType(ReductionType.NoReduction)

    def get_output(self, name: str, outputProjection: str = None):
        if not self.modelInitialised:
            raise RuntimeError("Model is not yet initialised.")

        if outputProjection is None:
            projOut = self.projSim
        else:
            projOut = ProjectionParameters.from_proj4(outputProjection)

        if name in self.outputRasters:
            out = self.outputRasters[name]
            out = project_raster(out, self.projSim, projOut, self.resolutionMeters)
            return out

        return None

    def get_arrival(self, outfile: str = None, write: bool = False,
                         outputProjection: str = None,
                         parameter: Union[str, Dict] = None):
        if not self.modelInitialised:
            raise RuntimeError("Model is not yet initialised.")

        if outputProjection is None:
            projOut = self.projSim
        else:
            projOut = ProjectionParameters.from_proj4(outputProjection)

        arrival = self.solver.getArrival()
        arrival.name = "arrival"
        arrival.setInterpolationType(RasterInterpolationType.Bilinear)
        arrival = project_raster(arrival, self.projSim,
                                             projOut, self.resolutionMeters)
        if arrival.name != "arrival":
            arrival.name = "arrival"

        if write:
            if outfile is not None:
                if isinstance(outfile, str):
                    if pth.exists(pth.dirname(outfile)):
                        if parameter is not None:
                            if isinstance(parameter, dict):
                                arrival.write(outfile, json.dumps(parameter))
                            elif isinstance(parameter, str):
                                arrival.write(outfile, parameter)
                        else:
                            arrival.write(outfile, "")
                    else:
                        raise ValueError("directory %s doesnt exist" % pth.dirname(outfile))
        else:
            return arrival

    def get_distance(self, outfile: str = None, write: bool = False,
                     outputProjection: str = None,
                     parameter: Union[str, dict] = None):
        if not self.modelInitialised:
            raise RuntimeError("Model is not yet initialised.")

        if outputProjection is None:
            projOut = self.projSim
        else:
            projOut = ProjectionParameters.from_proj4(outputProjection)

        distance = self.solver.getDistance()
        distance.name = "distance"
        distance = project_raster(distance, self.projSim,
                                            projOut, self.resolutionMeters)
        if distance.name != "distance":
            distance.name = "distance"

        if write:
            if outfile is not None:
                if isinstance(outfile, str):
                    if pth.exists(pth.dirname(outfile)):
                        if parameter is not None:
                            if isinstance(parameter, dict):
                                distance.write(outfile, json.dumps(parameter))
                            elif isinstance(parameter, str):
                                distance.write(outfile, parameter)
                        else:
                            distance.write(outfile, "")
                    else:
                        raise ValueError("directory %s doesn't exist" % pth.dirname(outfile))
        else:
            return distance

    def get_classification(self, outfile: str = None, write: bool = False,
                           outputProjection: str = None,
                           parameter: Union[Dict, str] = None):
        if not self.modelInitialised:
            raise RuntimeError("Model is not yet initialised.")

        if outputProjection is None:
            projOut = self.projSim
        else:
            projOut = ProjectionParameters.from_proj4(outputProjection)

        spark_class = self.solver.getClassification()
        spark_class.name = "classification"
        spark_class = project_raster(spark_class, self.projSim,
                                                 projOut, self.resolutionMeters)
        if spark_class.name != "classification":
            spark_class.name = "classification"

        if write:
            if outfile is not None:
                if isinstance(outfile, str):
                    if pth.exists(pth.dirname(outfile)):
                        if parameter is not None:
                            if isinstance(parameter, str):
                                spark_class.write(outfile, parameter)
                            elif isinstance(parameter, dict):
                                spark_class.write(outfile, json.dumps(parameter))
                        else:
                            spark_class.write(outfile, "")
                    else:
                        raise ValueError("directory %s doesn't exist" % pth.dirname(outfile))
        else:
            return spark_class

    def get_spot_fire_distribution(self, outfile: str = None, write: bool = False,
                                   outputProjection: str = None,
                                   parameter: Union[Dict, str] = None):
        if not self.modelInitialised:
            raise RuntimeError("Model is not yet initialised.")

        if 'firebrands' not in self.subModels:
            return None

        # Calculate spot fire distribution
        dist = self.spotFires.mapDistance(self.resolutionMeters, f'''
            REAL rb = exp(-dot(d, d)/pow({self.resolutionMeters}, 2));
            output = isValid_REAL(output) ? output+rb : rb;
        ''', bounds=self.solver.getClassification().getBounds())

        if outputProjection is not None:
            projOut = ProjectionParameters.from_proj4(outputProjection)
            dist = project_raster(dist, self.projSim, projOut,
                self.resolutionMeters)

        if write:
            if outfile is not None:
                if isinstance(outfile, str):
                    if pth.exists(pth.dirname(outfile)):
                        if parameter is not None:
                            if isinstance(parameter, str):
                                dist.write(outfile, parameter)
                            elif isinstance(parameter, dict):
                                dist.write(outfile, json.dumps(parameter))
                        else:
                            dist.write(outfile, "")
                    else:
                        raise ValueError("directory %s doesn't exist" % pth.dirname(outfile))
        else:
            return dist

    def get_isochrones(self, output_interval: Union[Integral, Real] = 3600.0) -> "Vector":
        """Get the isochrones from spark model output.

        Parameters
        ----------
        output_interval : Union[Integral, Real], optional
            the time interval for the isochrones, by default 3600.0

        Returns
        -------
        Vector
            an instance of geostack Vector object

        Raises
        ------
        RuntimeError
            Model is not yet initialised.
        TypeError
            Output interval should be numeric.
        ValueError
            Output_interval cannot be greater model duration
        ValueError
            Output interval cannot be greater model duration
        """

        if not self.modelInitialised:
            raise RuntimeError("Model is not yet initialised.")

        if not isinstance(output_interval, (int, float)):
            raise TypeError("Output interval should be numeric.")

        if output_interval > self.durationSeconds:
            raise ValueError("output_interval cannot be greater model duration")

        max_time = min(self.durationSeconds, self.solver_time)

        if output_interval > max_time:
            raise ValueError("Output interval cannot be greater model duration.")

        isochroneValues = np.arange(0.0, max_time, output_interval)
        isochroneVector = self.get_arrival().vectorise(isochroneValues,
                                                       2.0*self.durationSeconds)

        if isochroneVector.hasData():

            # Project to GeoJSON standard of EPSG:4326
            proj_EPSG4326 = ProjectionParameters.from_proj4(
                "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
            isochroneVector = isochroneVector.convert(proj_EPSG4326)

        return isochroneVector

    def get_metrics(self, metric: str = None):

        # Return area
        if metric == 'area':
            return { "area": self.solver.parameters.area }

        # Return elliptical fit to data
        elif metric == 'ellipse':
            max_time = min(self.durationSeconds, self.solver_time)
            isochroneVector = self.get_arrival().vectorise(max_time, 2.0*self.durationSeconds)
            if isochroneVector.hasData():

                # Create arrays
                N = isochroneVector.getVertexSize()
                x = np.empty(N)
                y = np.empty(N)
                for i in range(0, N):
                    c = isochroneVector.getCoordinate(i)
                    x[i] = c.p
                    y[i] = c.q

                # Centre data
                xmean, ymean = x.mean(), y.mean()
                x -= xmean
                y -= ymean

                # SVD
                u, s, v = np.linalg.svd(np.stack((x, y)))
                t = np.sqrt(2/N) * u.dot(np.diag(s))

                # Return array of mean with ellipse axis vectors
                return( {
                    "ellipse_centre": [xmean, ymean],
                    "ellipse_axis_a": [t[0][0], t[1][0]],
                    "ellipse_axis_b": [t[0][1], t[1][1]] } )

    def write_isochrones(self,
                         output_interval: Union[Integral, Real] = None,
                         outfile: str = None,
                         format: str="geojson",
                         type: str = "lines"):

        # Create isochrones
        isochroneVector = self.get_isochrones(output_interval=output_interval)

        # Update type
        if type != "lines" and type != "Lines":
            if type == "points" or type == "Points":
                isochroneVector = isochroneVector.convert(GeometryType.Point)
            elif type == "polygons" or type == "Polygons":
                isochroneVector = isochroneVector.convert(GeometryType.Polygon)
            else:
                raise TypeError(f"Type '{type}' not supported.")

        # Write isochrone file
        with open(outfile, 'w') as fp:
            if format == "geojson":
                fp.write(vectorToGeoJson(isochroneVector))
            elif format == "geowkt":
                fp.write(vectorToGeoWKT(isochroneVector))
            else:
                raise ValueError(f"format {format} is not valid")

    def get_vector_output(self,
                          name: str,
                          value: Union[Integral, Real],
                          type: str = "lines"):

        if not self.modelInitialised:
            raise RuntimeError("Model is not yet initialised.")

        if not isinstance(value, (int, float)):
            raise TypeError("Output interval should be numeric.")

        outputVector = Vector()
        if name in self.outputRasters:

            # Create isolines
            outputVector = self.outputRasters[name].vectorise(value,
                2.0*self.outputRasters[name].max()) # Max is expensive call

            if outputVector.hasData():

                # Update type
                if type != "lines" and type != "Lines":
                    if type == "points" or type == "Points":
                        outputVector = outputVector.convert(GeometryType.Point)
                    elif type == "polygons" or type == "Polygons":
                        outputVector = outputVector.convert(GeometryType.Polygon)
                    else:
                        raise TypeError(f"Type '{type}' not supported.")

                # Project to GeoJSON standard of EPSG:4326
                proj_EPSG4326 = ProjectionParameters.from_proj4(
                    "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
                outputVector = outputVector.convert(proj_EPSG4326)

        else:
            raise TypeError(f"Output '{name}' not found.")

        return outputVector

    def get_spot_fires(self) -> "Vector":
        """get the spot fire locations from the Spark model simulation.

        Returns
        -------
        Vector
            an instance of geostack vector object
        """

        # Return spot fires
        return self.spotFires

    def get_firebrands(self) -> "Vector":
        """get the firebrand paths from the Spark model simulation.

        Returns
        -------
        Vector
            an instance of geostack vector object
        """

        # Return spot fires
        return self.firebrandPaths

    def write_spot_fires(self, outfile: str = None, format: str = "geojson"):

        # Output spot fires
        if 'firebrands' in self.subModels:
            with open(outfile, 'w') as fp:
                if format == "geojson":
                    fp.write(vectorToGeoJson(self.spotFires))
                elif format == "geowkt":
                    fp.write(vectorToGeoWKT(self.spotFires))
                else:
                    raise ValueError(f"format {format} is not valid")

    def get_met_history(self,
                        outputScript: str = "",
                        outputs: List[str] = None,
                        step: int = 1):

        # Check for initialisation
        if not self.modelInitialised:
            logger.error("Spark model is not initialized.")
            raise RuntimeError("Spark model is not initialized.")

        # Check parameters
        if step <= 0:
            logger.error("Step size must be greater than zero.")
            raise RuntimeError("Step size must be greater than zero.")
        hasNoOutputs = outputs is None or len(outputs) == 0

        # Check gridded data
        metGriddedTypes = {}
        if self.metLayers is not None and len(self.metLayers) != 0:
            for metType, metVar in self.metVariables.items():
                metGriddedTypes[metType] = metVar

        # Check series data
        metSeriesTypes = {}
        if self.series is not None and self.series:
            for metType, metVar in self.seriesVariables.items():
                if metVar in self.series:
                    metSeriesTypes[metType] = metVar

        # Calculate spacings
        startEpochSeconds = self.startEpochMilliseconds/1000.0
        currentEpochSeconds = self.solver.getEpochMilliseconds()/1000.0
        nz = int(np.ceil((currentEpochSeconds-startEpochSeconds)/3600.0))+1
        delta = max(self.simulationsDims.ex-self.simulationsDims.ox,
                    self.simulationsDims.ey-self.simulationsDims.oy)

        # Create output Rasters  and script
        history = {}
        vectorScript = ''
        vectorScriptRasters = []
        for metType in {**metGriddedTypes, **metSeriesTypes}.keys():

            # Only include raster types included in outputs
            if hasNoOutputs or metType in outputs:

                # Create history Raster
                history[metType] = Raster(name = f"{metType}_history")
                history[metType].init(
                    nx = self.simulationsDims.nx/step,
                    ny = self.simulationsDims.ny/step,
                    nz = nz,
                    hx = self.simulationsDims.hx*step,
                    hy = self.simulationsDims.hy*step,
                    hz = 3600.0,
                    ox = self.simulationsDims.ox,
                    oy = self.simulationsDims.oy,
                    oz = startEpochSeconds)
                history[metType].setProjectionParameters(self.projSim)

                # Update vector script
                vectorScript += f"{metType} = {metType}_history;\n"
                vectorScriptRasters.append(history[metType])

        # Append any user-defined script
        vectorScript += '\n' + outputScript

        # Populate history Rasters
        script = ""
        scriptRasters = []
        dummyMetRasters = {}
        dummySeriesRasters = {}

        # Create mapping scripts and list
        for metType in metGriddedTypes:

            # Only include raster types included in outputs
            if hasNoOutputs or metType in outputs:

                # Get met item
                name = self.metRasterName[metType]
                item = None
                for layer in self.metLayers:
                    if layer['name'] == name:
                        item = layer
                        break
                if item is None:
                    logger.error(f"Cannot find met layer '{name}'")
                    raise RuntimeError(f"Cannot find met layer '{name}'")

                # Get gridded indexes
                iStart = self.metRasters[name].raster.getIndexFromTime(startEpochSeconds)
                if iStart > 0:
                    iStart -= 1 # Pad one time step backwards
                iEnd = self.metRasters[name].raster.getIndexFromTime(startEpochSeconds+3600.0*nz)
                if (iEnd - iStart < 2) and (iStart + 2 < self.metRasters[name].raster.getMaximumTimeIndex()-1):
                    iEnd = iStart + 2 # Pad two time steps forwards

                # Get correct handler
                backend, thredds = MetRaster.get_raster_handler(item['source'])

                # Create raster layer
                dummyMetRasters[metType] = MetRaster(RasterFile(name=name,
                                                             filePath=item['source'],
                                                             variable_map=self.gribRasterMap[name],
                                                             backend=backend,), thredds, slice(iStart, iEnd, 1))

                # Apply projection
                if 'projection' in item:
                    dummyMetRasters[metType].setProjectionParameters(item['projection'])

                # Set interpolation
                if item['type'] == 'wind_direction':
                    dummyMetRasters[metType].raster.setInterpolationType(RasterInterpolationType.Nearest)
                else:
                    dummyMetRasters[metType].raster.setInterpolationType(RasterInterpolationType.Bilinear)

                # Update script and script raster list
                script += f"{metType}_history = {name}*{self.metConversion.get(metType, 1)}+{self.metOffset.get(metType, 0)};\n"
                scriptRasters.append(history[metType])
                scriptRasters.append(dummyMetRasters[metType].raster)

        # Create dummy Rasters and script
        for metType, metVar in metSeriesTypes.items():

            # Only include raster types included in outputs
            if hasNoOutputs or metType in outputs:

                # Create dummy Raster
                dummySeriesRasters[metType] = Raster(name = f"{metType}")
                dummySeriesRasters[metType].init(
                    nx = 1,
                    ny = 1,
                    nz = nz,
                    hx = 2*delta,
                    hy = 2*delta,
                    hz = 3600.0,
                    ox = 0.5*(self.simulationsDims.ox+self.simulationsDims.ex)-delta,
                    oy = 0.5*(self.simulationsDims.oy+self.simulationsDims.ey)-delta,
                    oz = startEpochSeconds)
                dummySeriesRasters[metType].setProjectionParameters(self.projSim)

                for t in range(0, nz):
                    seriesTime = self.startEpochMilliseconds+3600000*t
                    dummySeriesRasters[metType].setCellValue(self.series[metVar].get(seriesTime), 0, 0, t)

                # Update script and script raster list
                script += f"{metType}_history = {metType};\n"
                scriptRasters.append(history[metType])
                scriptRasters.append(dummySeriesRasters[metType])

        # Create output
        centres = None

        # Map met data
        if len(scriptRasters) == 0:
            logger.warn(f"No data in met history for outputs '{outputs}'.")
            warnings.warn(f"No data in met history for outputs '{outputs}'.", RuntimeWarning)
        else:

            # Run mapping script
            runScript(script, scriptRasters)

            if len(history) > 0:

                # Create vector of cell centres
                centres = history[next(iter(history))].cellCentres()
                if centres.hasData():

                    # Add properties
                    if hasNoOutputs:
                        for metType in metGriddedTypes:
                            centres.addProperty(metType)
                        for metType in metSeriesTypes:
                            centres.addProperty(metType)
                    elif len(outputs) > 0:
                        for output in outputs:
                            centres.addProperty(output)

                    # Map to vector
                    runVectorScript(vectorScript, centres, vectorScriptRasters, parameter = VectorOrdering.Unordered)

                    # Project to GeoJSON standard of EPSG:4326
                    proj_EPSG4326 = ProjectionParameters.from_proj4(
                        "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
                    centres = centres.convert(proj_EPSG4326)

        return centres

    @property
    def solver_time(self):
        return self.solver.parameters.time

    @property
    def solver_area(self):
        return self.solver.parameters.area

    def __getattr__(self, attr: str):
        value = self.__dict__.get(attr)
        if not value:
            # Raise AttributeError if attribute value not found.
            logger.error(f'{self.__class__.__name__}.{attr} is invalid.')
            raise AttributeError(f'{self.__class__.__name__}.{attr} is invalid.')

        # Return attribute value.
        return value

    def __repr__(self):
        return "<class 'geostack.%s.%s'>" % (self.__module__,
                                     self.__class__.__name__)

def jaccard_score(simulatedExtent, recordedExtent, outFile=None,):

    # set reduction type
    simulatedExtent.setReductionType(ReductionType.Count)
    recordedExtent.setReductionType(ReductionType.Count)

    # run script to compare
    comparison = runScript('''
        if (isValid_REAL(simulationExtent) && isValid_REAL(recordedExtent))
            output = 1;
        ''', [simulatedExtent, recordedExtent], ReductionType.Count)

    # Calculate score
    jaccard = comparison.reduceVal / (recordedExtent.reduceVal +
                                      simulatedExtent.reduceVal -
                                      comparison.reduceVal)

    # Write comparison file
    if outFile is not None and outFile != '':
        comparison = runScript('''
            int c = 0;
            if (isValid_REAL(simulationExtent))
                c |= 0x01;
            if (isValid_REAL(recordedExtent))
                c |= 0x02;
            output = c;
            ''', [simulatedExtent, recordedExtent])

        # Write comparison
        comparison.write(outFile)
    return jaccard

