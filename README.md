# processing-imagestream-images
Creating a pipeline for importing and formatting ImageStream images for neural networks in R.

# Background
This pipeline uses an annotated ImageStream dataset consisting of two classes: ciliated cells, or unciliated cells.
Each image has two brightfield and a darkfield channel, collected at 60x on the ImageStream MkII at the Francis Crick Institute.

# Pipeline
The images have already been combined into 3-channel tifs in the "ciliated_cells.zip" file. If you unzip this directory you will see separate sub-directories called "train", "validate", and "test". Within each of these are "ciliated" and "unciliated" directories containing the images. This is the format keras expects.

To get the images into this format, they were manually tagged in IDEAS software and exported as unpadded, 8-bit ome.tifs. A FIJI macro "merge-channels.ijm" was used to iterate over each cell and merge its 3 image channels into a single tif

# R scripts
This repo contains various R scripts for analysing the data in different ways. The approach of each analysis should be pretty clear from the name of the script.
