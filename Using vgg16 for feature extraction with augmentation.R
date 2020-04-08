# Load packages -----------------------------------------------------------

library(keras)
library(tidyverse)

# Defining the file paths -------------------------------------------------

base_dir <- "~/Documents/Stats/Neural nets/ciliated_cells"
train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")

# Loading in the vgg16 model ----------------------------------------------

conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

# Defining the model architecture -----------------------------------------

model <- keras_model_sequential() %>% 
  conv_base %>% 
  layer_flatten() %>% 
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

# Freezing the weights of the convolutional base --------------------------

freeze_weights(conv_base)

# Defining the data generators --------------------------------------------

train_datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 60,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

test_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

# Compiling and fitting the model -----------------------------------------

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 2e-5),
  metrics = c("accuracy")
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 22,
  epochs = 100,
  validation_data = validation_generator,
  validation_steps = 9
)

plot(history) + 
  geom_point(shape = 21, col = "black", aes(fill = data)) +
  scale_fill_brewer(palette = "Set1") +
  scale_colour_brewer(palette = "Set1") +
  facet_wrap(~ metric) +
  coord_cartesian(ylim = c(0, 1)) +
  theme_bw()

ggsave("Pretrained VGG16 conv base with augmentation.pdf", 
       width = 10, height = 3)