# Load packages -----------------------------------------------------------

library(keras)
library(tidyverse)

# Defining the file paths -------------------------------------------------

base_dir <- "ciliated_cells"
train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")

# Loading in the vgg16 model ----------------------------------------------

conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

# Freezing the weights of the convolutional base --------------------------

freeze_weights(conv_base)
unfreeze_weights(conv_base, from = "block3_conv1")

# Adding a densley-connected classifier onto the convolutional base -------

model <- keras_model_sequential() %>% 
  conv_base %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

# Configure data augmentation generator -----------------------------------

train_datagen = image_data_generator(rescale = 1/255)

test_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(150, 150),
  batch_size = 22,
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 18,
  class_mode = "binary"
)

test_generator = flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 30,
  class_mode = NULL,  # only data, no labels
  shuffle= FALSE)  # keep data in same order as labels

# Compiling and fitting the model -----------------------------------------

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 2e-5),
  metrics = c("accuracy")
)

checkpoint_dir <- "checkpoints"
dir.create(checkpoint_dir, showWarnings = FALSE)
filepath <- file.path(checkpoint_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5")

# Create checkpoint callback
cp_callback <- callback_model_checkpoint(
  filepath = filepath,
  save_weights_only = TRUE,
  verbose = 1
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 4,
  epochs = 10,
  validation_data = validation_generator,
  validation_steps = 2,
  callbacks = list(cp_callback)
)

plot(history) + 
  geom_point(shape = 21, col = "black", aes(fill = data)) +
  scale_fill_brewer(palette = "Set1") +
  scale_colour_brewer(palette = "Set1") +
  #geom_smooth() +
  facet_wrap(~ metric) +
  coord_cartesian(ylim = c(0, 1)) +
  theme_bw()

ggsave("Fine-tuned VGG16 conv base.pdf", width = 10, height = 3)

# Load the model weights at epoch 7 ---------------------------------------

model7 <- load_model_weights_hdf5(model, "checkpoints/weights.07-0.06.hdf5")

save_model_hdf5(model7, "Fine_tune_vgg16.hdf5")

# Using the model to train predictions on the test set --------------------

train_generator$class_indices 

probabilities = predict_generator(model7, test_generator, steps = 1, workers = 4)

# Evaluating model fit on test set ----------------------------------------

predicted_class <- if_else(probabilities > 0.5, 1, 0)

true_class <- test_generator$filenames %>% str_replace_all("\\/.*", "")

model_predictions <- tibble(
  Truth = true_class,
  Predicted = case_when(predicted_class == 0 ~ "ciliated", 
                        predicted_class == 1 ~ "unciliated"),
  Prob = probabilities
)

confusion <- table(model_predictions$Truth, model_predictions$Predicted)

accuracy <- (confusion[1, 1] + confusion [2, 2]) / sum(confusion)
precision <- confusion[1, 1] / (confusion[1, 1] + confusion[1, 2])
recall <-  confusion[1, 1] / (confusion[1, 1] + confusion[2, 1])
f1 <- 2 / ((1 / precision) + (1 / recall))

# Visualizing the model's layer outputs -----------------------------------
scratch_model <- load_model_hdf5("cats_vs_dogs_model.h5")

img_path <- paste0(base_dir, "/test/ciliated/ciliated_33971_3_colour.ome.tif")

img <- image_load(img_path, target_size = c(150, 150))

img_tensor <- img %>% 
  image_to_array() %>% 
  array_reshape(c(1, 150, 150, 3)) %>%
  `/`(255)

dim(img_tensor)

plot(as.raster(img_tensor[1,,,]))

layer_outputs <- lapply(scratch_model$layers[1:8], function(layer) layer$output)

activation_model <- keras_model(inputs = scratch_model$input, 
                                outputs = layer_outputs)

activations <- activation_model %>% predict(img_tensor)

plot_channel <- function(channel) {
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(channel), axes = FALSE, asp = 1,
        col = terrain.colors(12))
}

first_layer_activation <- activations[[1]]

dim(first_layer_activation)

plot_channel(first_layer_activation[1,,,1]) # visualize a single output

# Visualize the activations for all layers --------------------------------

dir.create("cell_activations")
image_size <- 58
images_per_row = 16

for (i in 1:8) {
  
  layer_activation <- activations[[i]]
  layer_name <- scratch_model$layers[[i]]$name
  
  n_features <- dim(layer_activation)[[4]]
  n_cols <- n_features %/% images_per_row
  
  png(paste0("cell_activations/", i, "_", layer_name, ".png"), 
      width = image_size * images_per_row, 
      height = image_size * n_cols)
  op <- par(mfrow = c(n_cols, images_per_row), mai = rep_len(0.02, 4))
  
  for (col in 0:(n_cols-1)) {
    for (row in 0:(images_per_row-1)) {
      channel_image <- layer_activation[1,,,(col*images_per_row) + row + 1]
      plot_channel(channel_image)
    }
  }
  
  par(op)
  dev.off()
}
