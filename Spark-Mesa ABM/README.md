# fire-suppression-ABM

The part that interacts spark with ABM data has been changed, and the code can now be run directly to view, using the following script to run the code：

`python spark_conn.py --configFile ./data/spark_abm.json`


## Reference
The spark part of the code mainly uses the code developed by the Spark team: https://gitlab.com/geostack-applications/spark/-/wikis/home

The rate-of-speed model is set mainly by referring to this model library: https://research.csiro.au/spark/resources/model-library/
