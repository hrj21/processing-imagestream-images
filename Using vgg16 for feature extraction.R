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

summary(conv_base)

# Performing feature extraction with existing convnet ---------------------

datagen <- image_data_generator(rescale = 1/255)

batch_size <- 2

extract_features <- function(directory, sample_count) {
  
  features <- array(0, dim = c(sample_count, 4, 4, 512))  
  labels <- array(0, dim = c(sample_count))
  
  generator <- flow_images_from_directory(
    directory = directory,
    generator = datagen,
    target_size = c(150, 150),
    batch_size = batch_size,
    class_mode = "binary"
  )
  
  i <- 0
  while(TRUE) {
    batch <- generator_next(generator)
    inputs_batch <- batch[[1]]
    labels_batch <- batch[[2]]
    features_batch <- conv_base %>% predict(inputs_batch)
    
    index_range <- ((i * batch_size)+1):((i + 1) * batch_size)
    features[index_range,,,] <- features_batch
    labels[index_range] <- labels_batch
    
    i <- i + 1
    if (i * batch_size >= sample_count)
      # Note that because generators yield data indefinitely in a loop, 
      # you must break after every image has been seen once.
      break
  }
  
  list(
    features = features, 
    labels = labels
  )
}

train <- extract_features(train_dir, 88)
validation <- extract_features(validation_dir, 36)
test <- extract_features(test_dir, 30)

# Reshape extracted features ----------------------------------------------

reshape_features <- function(features) {
  array_reshape(features, dim = c(nrow(features), 4 * 4 * 512))
}

train$features <- reshape_features(train$features)
validation$features <- reshape_features(validation$features)
test$features <- reshape_features(test$features)

# Define, compile, and train a densely-connected classifier ---------------

model <- keras_model_sequential() %>% 
  layer_dense(units = 64, activation = "relu", 
              input_shape = 4 * 4 * 512) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% fit(
  train$features, train$labels,
  epochs = 100,
  batch_size = 22,
  validation_data = list(validation$features, validation$labels)
)

plot(history) + 
  geom_point(shape = 21, col = "black", aes(fill = data)) +
  scale_fill_brewer(palette = "Set1") +
  scale_colour_brewer(palette = "Set1") +
  facet_wrap(~ metric) +
  coord_cartesian(ylim = c(0, 1)) +
  theme_bw()

ggsave("Pretrained VGG16 conv base.pdf", width = 10, height = 3)

# Using the model to train predictions on the test set --------------------

probabilities = predict(model, test$features, steps = 1, workers = 4)

# Evaluating model fit on test set ----------------------------------------

predicted_class <- if_else(probabilities > 0.5, 1, 0)

model_predictions <- tibble(
  Truth = case_when(test$labels == 0 ~ "ciliated",
                    test$labels == 1 ~ "unciliated"),
  Predicted = case_when(predicted_class == 0 ~ "ciliated", 
                        predicted_class == 1 ~ "unciliated"),
  Prob = probabilities
)

confusion <- table(model_predictions$Truth, model_predictions$Predicted)

accuracy <- (confusion[1, 1] + confusion [2, 2]) / sum(confusion)
precision <- confusion[1, 1] / (confusion[1, 1] + confusion[1, 2])
recall <-  confusion[1, 1] / (confusion[1, 1] + confusion[2, 1])
f1 <- 2 / ((1 / precision) + (1 / recall))