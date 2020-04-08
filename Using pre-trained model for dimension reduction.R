# Load packages -----------------------------------------------------------

library(keras)
library(tidyverse)
library(patchwork)

# Defining the file paths -------------------------------------------------

base_dir <- "ciliated_cells"
train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")

# Configure data augmentation generator -----------------------------------

datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(150, 150),
  batch_size = 22,
  class_mode = "binary",
  shuffle = FALSE
)

test_generator <- flow_images_from_directory(
  test_dir,
  datagen,
  target_size = c(150, 150),
  batch_size = 30,
  class_mode = "binary",
  shuffle = FALSE
)

# Load in saved model -----------------------------------------------------

model <- load_model_hdf5("Fine_tune_vgg16.hdf5")

# Extract output from dense layer before sigmoid --------------------------

final_dense_acts <- keras_model(inputs = model$input,
                                outputs = model$layers[[3]]$output)

dense_output <- predict_generator(final_dense_acts, 
                                  train_generator, steps = 4, workers = 4)

str(dense_output)

true_class <- train_generator$filenames %>% str_replace_all("\\/.*", "")

# Running UMAP on flattened, extracted features ----------------------------

library(umap)
library(Rtsne)

train_umap <- umap(dense_output)
train_tsne <- Rtsne(dense_output, perplexity = 20)
train_pca <- prcomp(dense_output)

dense_output_tib <- as_tibble(dense_output) %>%
  mutate(Truth = true_class,
         UMAP1 = train_umap$layout[, 1],
         UMAP2 = train_umap$layout[, 2],
         tSNE1 = train_tsne$Y[, 1],
         tSNE2 = train_tsne$Y[, 2],
         PCA1 = train_pca$x[, 1],
         PCA2 = train_pca$x[, 2])

p1 <- ggplot(dense_output_tib, aes(UMAP1, UMAP2, col = Truth)) +
  geom_point() +
  theme_bw() +
  coord_equal(ylim = c(-15, 15), xlim = c(-15, 15)) +
  theme(legend.position = "none")

p2 <- ggplot(dense_output_tib, aes(tSNE1, tSNE2, col = Truth)) +
  geom_point() +
  theme_bw() +
  coord_equal(ylim = c(-17, 17), xlim = c(-17, 17)) +
  theme(legend.position = "none")

p3 <- ggplot(dense_output_tib, aes(PCA1, PCA2, col = Truth)) +
  geom_point() +
  coord_equal(ylim = c(-10, 10), xlim = c(-10, 10)) +
  theme_bw() 

p1 + p2 + p3
ggsave("Dimension reduction of dense layer output.png", width = 10, height = 3)
