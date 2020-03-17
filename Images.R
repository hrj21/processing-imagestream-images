
# Load packages -----------------------------------------------------------

for(pkg in c("tidyverse", "imager", "jpeg", "ptw")) {
  if(!require(pkg, character.only = TRUE)){
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  } else(library(pkg, character.only = TRUE))
}

# # CREATE VECTOR OF FILEPATHS ---- ---------------------------------------

filenames <- list.files(path = "jpgs", pattern = "*.jpg", full.names = TRUE)

# Read images as matrices -------------------------------------------------

cell <- map(filenames, readJPEG)

cell[[1]]

# Visualize image data ----------------------------------------------------

par(mfrow = c(8, 9), mai = c(0, 0, 0, 0))

walk(cell, function(x) {
  image(x, useRaster = TRUE, axes = FALSE, col = grey(seq(0, 1, length = 2^16)))
  box(col = "red")
})

# Find largest dimensions among the images --------------------------------

max_dims <- map_df(cell, ~ data.frame(rows = dim(.)[1], cols = dim(.)[2])) %>%
  summarize(max_row = max(rows), max_col = max(cols))

# Pad all images to have the same dimensions ------------------------------
padded_cells <- map(cell, function(x) {
  padzeros(x, max_dims[1, 1] - dim(x)[1], side = "left") %>%
    t() %>%
    padzeros(max_dims[1, 2] - dim(x)[2], side = "left")
})

# Plot padded images ------------------------------------------------------
par(mfrow = c(8, 9), mai = c(0, 0, 0, 0))

walk(padded_cells, function(x) {
  image(x, useRaster = TRUE, axes = FALSE, col = grey(seq(0, 1, length = 2^16)))
  box(col = "red")
})

map_df(padded_cells, ~ data.frame(rows = dim(.)[1], cols = dim(.)[2]))
  
