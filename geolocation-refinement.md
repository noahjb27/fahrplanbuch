I need a flask application in which I can choose a dataset from data/processed/YEAR_SIDE/stops.csv.
YEAR and SIDE need to be selected.

The application needs to visualise the stops using their location which has this format: "52.42636276,13.37438168". The visualisation should be over a openstreetmap layer, probably with leaflet. I should be able drag the stations to new locations. These should then somehow be saved, I should be able to somehow track the station improvements.

I have some tif files which have been georeferenced in qgis. I want to be able to use the tif files in the visualisation as a layer.