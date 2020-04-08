# Load packages -----------------------------------------------------------

library(keras)
library(tidyverse)

# Defining the file paths -------------------------------------------------

base_dir <- "ciliated_cells"
train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")

# Defining the model architecture from scratch ----------------------------

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation="relu",
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")

# Define data generators --------------------------------------------------

datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(150, 150),
  batch_size = 22,
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  datagen,
  target_size = c(150, 150),
  batch_size = 18,
  class_mode = "binary"
)

test_generator = flow_images_from_directory(
  test_dir,
  datagen,
  target_size = c(150, 150),
  batch_size = 30,
  class_mode = NULL,  # only data, no labels
  shuffle = FALSE)  # keep data in same order as labels

# Compiling and fitting the model -----------------------------------------

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 2e-5),
  metrics = c("accuracy")
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 4,
  epochs = 100,
  validation_data = validation_generator,
  validation_steps = 2
)

plot(history) + 
  geom_point(shape = 21, col = "black", aes(fill = data)) +
  scale_fill_brewer(palette = "Set1") +
  scale_colour_brewer(palette = "Set1") +
  facet_wrap(~ metric) +
  coord_cartesian(ylim = c(0, 1)) +
  theme_bw()

ggsave("Convnet from scratch.pdf", width = 10, height = 3)

# Using the model to train predictions on the test set --------------------

train_generator$class_indices 

probabilities = predict_generator(model, test_generator, steps = 1, workers = 4)

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
