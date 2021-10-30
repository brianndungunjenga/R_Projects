
library(knitr)
knitr::opts_chunk$set(cache = TRUE, cache.lazy = FALSE,
                      message = FALSE, echo = TRUE, dpi = 180,
                      fig.width = 8, fig.height = 5)

# Explore The Data
library(tidyverse)
water_raw <- read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-05-04/water.csv")

water_raw %>% View()

water_raw %>%
  filter(country_name == "Sierra Leone",
         lat_deg > 0, lat_deg < 15, lon_deg < 0,
    status_id%in% c("y", "n")) %>%
  ggplot(aes(lon_deg, lat_deg, color = status_id)) +
  geom_point(alpha = 0.1) +
  coord_fixed() +
  guides(color = guide_legend(override.aes = list(alpha = 1)))

water <- water_raw %>%
  filter(country_name == "Sierra Leone",
         lat_deg > 0, lat_deg < 15, lon_deg < 0,
         status_id %in% c("y", "n")) %>%
  mutate(pay = case_when(str_detect(pay, "^No") ~ "no",
                         str_detect(pay, "^Yes") ~ "yes",
                         is.na(pay) ~ pay,
                         TRUE ~ "it's complicated")) %>%
  select(-country_name, -status, -report_date) %>%
  mutate_if(is.character, as.factor)

water %>%
  ggplot(aes(install_year, y = ..density.., fill = status_id)) +
  geom_histogram(position = "identity", alpha = 0.5) + 
  labs(fill = "Water Available?")

water %>%
  ggplot(aes(y = pay, fill = status_id)) +
  geom_bar(position = "fill") +
  labs(fill = "Water available?")

# Build a Model
library(tidymodels)

set.seed(123)
water_split <- initial_split(water, strata = status_id)
water_train <- training(water_split)
water_test <- testing(water_split)

set.seed(234)
water_folds <- vfold_cv(water_train, strata = status_id)
water_folds

library(themis)
ranger_recipe <- 
  recipe(formula = status_id ~ ., data = water_train) %>%
  update_role(row_id, new_role = "id") %>%
  step_unknown(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.03) %>%
  step_impute_linear(install_year) %>%
  step_downsample(status_id)

ranger_spec <- 
  rand_forest(trees = 1000) %>% 
  set_mode("classification") %>% 
  set_engine("ranger") 

ranger_workflow <- 
  workflow() %>% 
  add_recipe(ranger_recipe) %>% 
  add_model(ranger_spec) 

doParallel::registerDoParallel()
set.seed(74403)
ranger_tune <-
  fit_resamples(ranger_workflow,
            resamples = water_folds,
            control = control_resamples(save_pred = TRUE))
ranger_rs = ranger_tune

# Explore Results
collect_metrics(ranger_rs)

collect_predictions(ranger_rs) %>%
  group_by(id) %>%
  roc_curve(status_id, .pred_n) %>%
  autoplot()

# Confusion Matrix
conf_mat_resampled(ranger_rs, tidy = FALSE) %>%
  autoplot()

final_fitted <- last_fit(ranger_workflow, water_split)
collect_metrics(final_fitted)

collect_predictions(final_fitted) %>%
  conf_mat(status_id, .pred_class) %>%
  autoplot()

# Write Model to Disk to use later for prediction
# final_fitted$.workflow[[1]] %>% write_rds()

# Variable importance
library(vip)

imp_data <- ranger_recipe %>% prep() %>% bake(new_data = NULL)

ranger_spec %>%
  set_engine("ranger", importance = "permutation") %>%
  fit(status_id ~ ., data = imp_data) %>%
  vip(geom = "point")

imp_data %>%
  select(status_id, pay, water_tech, installer) %>%
  pivot_longer(pay:installer, names_to = "feature", values_to = "value") %>%
  ggplot(aes(y = value, fill = status_id)) +
  geom_bar(position = "fill") +
  facet_grid(rows = vars(feature), scales = "free_y", space = "free_y") +
  labs(fill = "Water Available?",
       x = "% of water sources", y = NULL) +
  scale_fill_brewer(type = "qual") + 
  scale_x_continuous(labels = scales::percent)

  
  




