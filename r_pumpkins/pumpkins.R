library(tidyverse)

pumpkins_raw <- readr::read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-10-19/pumpkins.csv')

pumpkins <-
  pumpkins_raw %>%
  separate(id, into = c("year", "type")) %>%
  filter(type == "P") %>%
  mutate(across(c(year, weight_lbs, ott, place), parse_number)) %>%
  select(weight_lbs, ott, year, place, gpc_site, country)
  
# EDA
pumpkins %>%
  filter(ott < 1e3, ott > 10) %>%
  ggplot(aes(ott, weight_lbs, color = place)) +
  geom_point(alpha = 0.1, size = 1.1) +
  labs(x = "over-the-top inches", y = "weight (lbs)") + 
  scale_color_viridis_c()


pumpkins %>%
  filter(ott < 1e3, ott > 10) %>%
  ggplot(aes(ott, weight_lbs)) +
  geom_point(alpha = 0.1, size = 1.1, color = "gray60") +
  geom_smooth(aes(color = factor(year)),
              method = "lm", formula = y ~ splines::bs(x, 3),
              se = FALSE, size = 1.5, alpha = 0.6) +
  labs(x = "over-the-top inches", y = "weight (lbs)") + 
  scale_color_viridis_d()

pumpkins %>%
  mutate(country = fct_lump(country, n=10),
         country = fct_reorder(country, weight_lbs)) %>%
  ggplot(aes(country, weight_lbs, color = country)) +
  geom_boxplot(outlier.color = NA) +
  geom_jitter(alpha = 0.1, width = 0.15) +
  labs(x = NULL, y = "weight (lbs)") +
  theme(legend.position = "none")


# Build a workflow
library(tidymodels)
set.seed(123)
pumpkin_split <-
  pumpkins %>%
  filter(ott < 1e3, ott > 20) %>%
  initial_split(strata = weight_lbs)

pumpkin_train <- training(pumpkin_split)
pumpkin_test <- testing(pumpkin_split)

set.seed(234)
pumpkin_folds <- vfold_cv(pumpkin_train, strata = weight_lbs)
pumpkin_folds

# A workflow allows us to combine multiple pre-processors and models and helps
# to determine optimum performance.

base_rec <-
  recipe(weight_lbs ~ ott + year + country + gpc_site,
         data = pumpkin_train) %>%
  step_other(country, gpc_site, threshold = 0.02)


ind_rec <-
  base_rec %>%
  step_dummy(all_nominal_predictors())

spline_rec <-
  ind_rec %>%
  step_bs(ott)

rf_spec <-
  rand_forest(trees = 1e3) %>%
  set_mode("regression") %>%
  set_engine("ranger")

mars_spec <-
  mars() %>%
  set_mode("regression") %>%
  set_engine("earth")

lm_spec <- linear_reg()

# A Workflow set combines a list of pre-processors with a list of models and 
# lets you handle them altogether.

pumpkin_set <-
  workflow_set(
    list(base_rec, ind_rec, spline_rec),
    list(rf_spec, mars_spec, lm_spec),
    cross = FALSE
  )

pumpkin_set

doParallel::registerDoParallel()
set.seed(2021)

pumpkin_rs <-
  workflow_map(
    pumpkin_set,
    "fit_resamples",
    resamples = pumpkin_folds
  )

pumpkin_rs


# Evaluate a Workflow Set
autoplot(pumpkin_rs)

collect_metrics(pumpkin_rs)

final_fit <-
  extract_workflow(pumpkin_rs, "recipe_3_linear_reg") %>%
  fit(pumpkin_train)

predict(final_fit, pumpkin_test)

tidy(final_fit)




