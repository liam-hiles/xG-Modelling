# LHiles xG Model Assignment ----

## Load Packages ----

library(tidyverse)
library(tidymodels)
library(ggplot2)
library(ggsoccer)
library(workflowsets)
library(finetune)
library(ranger)
library(xgboost)
library(stacks)
library(doParallel)
library(broom)
library(ggrepel)

## Load Data ----

original_data <- read_rds('~prem_event_data (1).rds')


## C.1 Inspect Data ----

# check all actions to find which ones count as shots
# SavedShot
# Goal
# MissedShots
# ShotOnPost
original_data %>% 
  filter(isShot == 1) %>% 
  distinct(action)

# check which columns interact with shots
# used to decide which predictors to use
original_data %>% 
  filter(isShot == 1) %>% 
  glimpse()

# check all positions
# will be used for positional groups
# 27 inc 'NA' (Managers)
# Goalkeepers
# GK
# Defenders
# CB, LCB, RCB, LB, RB, LWB, RWB
# Midfielders
# CDM, LCDM, RCDM, CM, LCM, RCM, LM, RM, CAM, LAM, RAM, LCAM, RCAM
# Forwards
# LW, RW, ST, LST, RST
# Managers / NA
# <NA>
original_data %>% 
  distinct(position)

# coordinates set-up
pitch_international
# length = 105
# width = 68
# goal width = 7.32.
# centre of goal = 34. (68/2)
# left_post = 30.34.  34 - (7.32/2) 
# right_post = 37.66. 34 + (7.32/2)


## C.2 Data Cleaning + Pre-Processing + Visual Check ----

# basic shots df
# filter for only shots
# removes own goals and penalties
# renames key columns
# 9718 rows
og_shots <- original_data %>% 
  filter(isShot == 1,
         q.own_goal == 0,
         q.penalty == 0) %>% 
  rename(shot_x = x,
         shot_y = y,
         assisted = q.assisted,
         direct_freekick = q.direct_freekick,
         from_corner = q.from_corner,
         header = q.head,
         left_foot = q.left_foot,
         right_foot = q.right_foot,
         regular_play = q.regular_play,
         big_chance = q.big_chance,
         fast_break = q.fast_break)

# pre-processing df
# creates distance and angle columns based on pitch coordinates
# angle is created based on euclidean distance 
# creates positional groups from position
# 9718 rows
shots_under_construction <- og_shots %>% 
  mutate(goal_x = 105,
         goal_y = 34,
         left_post_y = 30.34,
         right_post_y = 37.66,
         post_x = 105,
         goal = case_when(action == 'Goal' ~ 1,
                          TRUE ~ 0)) %>% 
  mutate(distance = sqrt((goal_x - shot_x)^2 + (goal_y - shot_y)^2)) %>% 
  mutate(angle = abs(atan2(left_post_y - shot_y, post_x - shot_x) - 
                       atan2(right_post_y - shot_y, post_x - shot_x)) * (180/pi)) %>% 
  mutate(position_group = case_when(
    position %in% c('GK') ~ 'GK',
    position %in% c('RB', 'LB', 'RCB', 'LCB', 'RWB', 'LWB', 'CB') ~ 'DEF',
    position %in% c('RCDM', 'LCDM', 'RAM', 'LAM', 'CAM', 'LCM', 'RCM', 'LM',
                    'CDM', 'RM', 'CM', 'LCAM', 'RCAM') ~ 'MID',
    position %in% c('ST', 'LST', 'RST', 'LW', 'RW' ) ~ 'ATT',
    TRUE ~ 'MISSING'
  )) 

# shot map to double check there are no outlying data points
shot_map <- shots_under_construction %>% 
  arrange(goal) %>% 
  ggplot(aes(x = shot_x, y = shot_y)) +
  annotate_pitch(
    dimensions = pitch_international,
    colour = 'black'
  ) +
  geom_point(
    aes(colour = factor(goal)),
    size = 2,
    alpha = 0.6
  ) +
  scale_colour_manual(
    values = c('0' = 'blue', '1' = 'red'),
    labels = c('0' = 'No Goal', '1' = 'Goal'),
    name = NULL
  ) +
  theme_pitch() +
  labs(title = 'All the Shots Taken in the Data')

## C.3 Key Variables + Formula + Metrics + Summary Tables ----

# predictor variables
# 13 predictors
ind_variables <- c('distance', 'angle', 'header', 'right_foot', 'left_foot', 
                   'big_chance', 'fast_break', 'direct_freekick', 'from_corner', 
                   'assisted', 'regular_play', 'minute', 'position_group')
# target variable
target_variable <- 'goal'

# combining variables to make formula
right_side_formula  <- paste(ind_variables, collapse = ' + ')
formula_complete <- as.formula(sprintf('%s ~ 1 + %s', target_variable, right_side_formula))

# key metrics
metrics <- metric_set(mn_log_loss, accuracy, roc_auc)

# final, clean df that only includes key variables
# 9718 shots
shots_complete <- shots_under_construction %>% 
  select(all_of(target_variable), all_of(ind_variables)) %>% 
  mutate(goal = factor(goal, levels = c(1, 0)))

# df showing the share of shots that were converted to goals
# 8739 not-goals 89.9%
# 979 goals 10.%
shots_share <- shots_complete %>% 
  count(goal) %>% 
  mutate(share = round(n / sum(n) * 100, 1))

# df showing the average distance and angle for goals vs not-goals
# 8739 not-goals
# 979 goals
avg_dist_angle <- shots_complete %>%
  group_by(goal) %>%
  summarise(
    mean_distance = round(mean(distance), 2),
    mean_angle = round(mean(angle), 2),
    n_shots = n()
  )

## C.4 Data Split + Cross Validation Folds ----

# set seed for reproducibility and create 80-20 training-test split
# stratified via target variable to ensure a proportional data split
set.seed(0304)
split <- initial_split(shots_complete, prop = 0.8, strata = 'goal')
train <- training(split)
test <- testing(split)

# same technique as above but for cross validation folds
set.seed(0304)
five_folds <- vfold_cv(train, v = 5, strata = 'goal')


## C.5 Recipes ----
# creates dummy variables for position groups for 2 recipes
# all have linear combinations removed as a precaution

recipe_glm <- recipe(formula_complete, data = train) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_lincomb(all_numeric_predictors())

recipe_tree <- recipe(formula_complete, data = train) %>%
  step_lincomb(all_numeric_predictors())

recipe_xgb <- recipe(formula_complete, data = train) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_lincomb(all_numeric_predictors())


## C.6 Model Specifications ----
# creates model specifications, all using classification
# 2 seperate xgboost models
# 5 total, 1 linear, 2 tree, 2 XGBoost

spec_glmnet <- logistic_reg(
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine('glmnet', standardize = FALSE, intercept = TRUE) %>%
  set_mode('classification')

spec_dec_tree <- decision_tree(
  cost_complexity = tune(),
  tree_depth      = tune(),
  min_n           = tune()
) %>%
  set_engine('rpart') %>%
  set_mode('classification')

spec_rf <- rand_forest(
  trees = 1000,
  mtry  = tune(),
  min_n = tune()
) %>%
  set_engine(
    'ranger', importance = 'impurity'
  ) %>%
  set_mode('classification')

spec_xgb_basic <- boost_tree(
  trees      = tune(),
  learn_rate = tune()
) %>%
  set_engine('xgboost') %>%
  set_mode('classification')

spec_xgb_advanced <- boost_tree(
  trees          = tune(),
  tree_depth     = tune(),
  learn_rate     = tune(),
  min_n          = tune(),
  loss_reduction = tune(),
  sample_size    = tune(),
  mtry           = tune()
) %>%
  set_engine('xgboost') %>%
  set_mode('classification')


## C.7 Predictor Mapping ----
# used later when setting tuning parameters 
# glm used for vip visual later

rec_glm_prep <- prep(recipe_glm, training = train, verbose = FALSE)
p_glm <- bake(rec_glm_prep, new_data = train) %>% 
  select(-goal) %>% 
  ncol() %>% 
  as.integer()
# 15

rec_tree_prep <- prep(recipe_tree, training = train, verbose = FALSE)
p_tree <- bake(rec_tree_prep, new_data = train) %>%
  select(-goal) %>%
  ncol() %>%
  as.integer()
# 13

rec_xgb_prep <- prep(recipe_xgb, training = train, verbose = FALSE)
p_xgb <- bake(rec_xgb_prep, new_data = train) %>%
  select(-goal) %>%
  ncol() %>%
  as.integer()
# 15


## C.8 Tuning Parameters V1 ----
# assigning an inital set of tuning parameters
# will later have a more fine set for highest performing models

params_glmnet <- workflow() %>% 
  add_recipe(recipe_glm) %>% 
  add_model(spec_glmnet) %>% 
  extract_parameter_set_dials() %>% 
  update(
    penalty = penalty(range = c(-4, 1)),
    mixture = mixture(range = c(0, 1))
  )

params_dec_tree <- workflow() %>% 
  add_recipe(recipe_tree) %>% 
  add_model(spec_dec_tree) %>% 
  extract_parameter_set_dials() %>% 
  update(
    cost_complexity = cost_complexity(range = c(-4, -1)),
    min_n           = min_n(range = c(10L, 150L)),
    tree_depth      = tree_depth(range = c(2L, 15L))
  )

params_rf <- workflow() %>% 
  add_recipe(recipe_tree) %>% 
  add_model(spec_rf) %>% 
  extract_parameter_set_dials() %>% 
  update(
    mtry  = mtry(range = c(2L, p_tree)),
    min_n = min_n(range = c(5L, 100L))
  )

params_xgb_basic <- workflow() %>% 
  add_recipe(recipe_xgb) %>% 
  add_model(spec_xgb_basic) %>% 
  extract_parameter_set_dials() %>% 
  update(
    trees      = trees(range = c(200L, 4000L)),
    learn_rate = learn_rate(range = c(-3, -0.5))
  )

params_xgb_advanced <- workflow() %>% 
  add_recipe(recipe_xgb) %>% 
  add_model(spec_xgb_advanced) %>% 
  extract_parameter_set_dials() %>% 
  update(
    trees          = trees(range = c(200L, 4000L)),
    tree_depth     = tree_depth(range = c(2L, 10L)),
    learn_rate     = learn_rate(range = c(-3, -0.5)),
    min_n          = min_n(range = c(2L, 40L)),
    loss_reduction = loss_reduction(range = c(-4, 1)),
    sample_size    = sample_prop(range = c(0.5, 1.0)),
    mtry           = mtry(range = c(2L, p_xgb))
  )


## C.9 First WorkFlow Set + Setting Up Cores ----

#combines all workflows into a set for racing
# ids on the left, recipes + model specs on the right
wf_tuned <- workflow_set(
  preproc = list(
    glmnet = recipe_glm,
    dec_tree = recipe_tree,
    random_forest = recipe_tree,
    xgb_basic = recipe_xgb,
    xgb_advanced = recipe_xgb
  ),
  models = list(
    glmnet = spec_glmnet,
    dec_tree = spec_dec_tree,
    random_forest = spec_rf,
    xgb_basic = spec_xgb_basic,
    xgb_advanced = spec_xgb_advanced
  ),
  cross = FALSE
)

# counts number of cores available
# uses cores - 1 as a precaution
cores <- detectCores(all.tests = FALSE, logical = TRUE)
n_cores <- cores - 1 
cl <- makePSOCKcluster(n_cores) 
registerDoParallel(cl)


## C.10 First Tuning Race ----

# setting up the race controls
race_ctrl <- control_race(
  verbose       = TRUE, 
  verbose_elim  = TRUE,   
  save_workflow = TRUE,   
  save_pred     = TRUE, 
  randomize     = FALSE,
  burn_in       = 3,       
  parallel_over = 'everything'
)

# racing the workflows
set.seed(0304)
wf_tuned_res <- wf_tuned %>%
  workflow_map(
    fn        = 'tune_race_anova',
    seed      = 0304,
    resamples = five_folds,
    grid      = 600,
    metrics   = metrics,
    control   = race_ctrl
  )

# ending the parallel processing
stopCluster(cl)

# saving and loading the results of the first workflow set
#saveRDS(wf_tuned_res, file = 'wf_tuned_res.rds')
#wf_tuned_res <- readRDS('~wf_tuned_res.rds')


# First Pass Model Rankings

tuned_rank <- rank_results(
  wf_tuned_res,
  rank_metric = 'mn_log_loss',
  select_best = TRUE
)

cv_rank <- tuned_rank %>% 
  filter(.metric == 'mn_log_loss') %>%
  arrange(mean) %>%
  mutate(rank = row_number()) %>%
  select(rank, wflow_id, model, preprocessor, mean, std_err)

# Best Model Parameters

best_params_glmnet <- wf_tuned_res %>%
  extract_workflow_set_result('glmnet_glmnet') %>%
  collect_metrics() %>%
  filter(.metric == 'mn_log_loss') %>%
  arrange(mean) %>%
  head(10)

best_params_rf <- wf_tuned_res %>%
  extract_workflow_set_result('random_forest_random_forest') %>%
  collect_metrics() %>%
  filter(.metric == 'mn_log_loss') %>%
  arrange(mean) %>%
  head(10)

best_params_xgb_advanced <- wf_tuned_res %>%
  extract_workflow_set_result('xgb_advanced_xgb_advanced') %>%
  collect_metrics() %>%
  filter(.metric == 'mn_log_loss') %>%
  arrange(mean) %>%
  head(10)

## C.11 Second WorkFlow Set ----

#new tuning ranges based on the best results from the first pass
fine_glmnet <- workflow() %>% 
  add_recipe(recipe_glm) %>% 
  add_model(spec_glmnet) %>% 
  extract_parameter_set_dials() %>% 
  update(
    penalty = penalty(range = c(-4, -2)),
    mixture = mixture(range = c(0.02, 0.25))
  )

fine_rf <- workflow() %>% 
  add_recipe(recipe_tree) %>% 
  add_model(spec_rf) %>% 
  extract_parameter_set_dials() %>% 
  update(
    mtry  = mtry(range = c(2L, p_tree)),
    min_n = min_n(range = c(2L, 10L))
  )
  
fine_xgboost_advanced_A <- workflow() %>% 
  add_recipe(recipe_xgb) %>% 
  add_model(spec_xgb_advanced) %>% 
  extract_parameter_set_dials() %>% 
  update(
    trees          = trees(range = c(100L, 500L)),
    tree_depth     = tree_depth(range = c(1L, 4L)),
    learn_rate     = learn_rate(range = c(-2, -1.2)),
    min_n          = min_n(range = c(2L, 8L)),
    loss_reduction = loss_reduction(range = c(-2, -1)),
    sample_size    = sample_prop(range = c(0.5, 0.9)),
    mtry           = mtry(range = c(2L, p_xgb))
  )

fine_xgboost_advanced_B <- workflow() %>% 
  add_recipe(recipe_xgb) %>% 
  add_model(spec_xgb_advanced) %>% 
  extract_parameter_set_dials() %>% 
  update(
    trees          = trees(range = c(1000L, 2000L)),
    tree_depth     = tree_depth(range = c(2L, 5L)),
    learn_rate     = learn_rate(range = c(-3.5, -2.2)),
    min_n          = min_n(range = c(4L, 10L)),
    loss_reduction = loss_reduction(range = c(-10, -1)),
    sample_size    = sample_prop(range = c(0.7, 0.95)),
    mtry           = mtry(range = c(2L, p_xgb))
  )

# second workflow with the new models and new tuning ranges
fine_wf_tuned <- workflow_set(
  preproc = list(
    glmnet = recipe_glm,
    random_forest = recipe_tree,
    xgb_advanced_A = recipe_xgb,
    xgb_advanced_B = recipe_xgb
  ),
  models = list(
    glmnet = spec_glmnet,
    random_forest = spec_rf,
    xgb_advanced_A = spec_xgb_advanced,
    xgb_advanced_B = spec_xgb_advanced
  ),
  cross = FALSE
) %>%
  option_add(param_info = fine_glmnet, id = 'glmnet_glmnet') %>%
  option_add(param_info = fine_rf, id = 'random_forest_random_forest') %>%
  option_add(param_info = fine_xgboost_advanced_A, id = 'xgb_advanced_A_xgb_advanced_A') %>%
  option_add(param_info = fine_xgboost_advanced_B, id = 'xgb_advanced_B_xgb_advanced_B')


# counts number of cores available
# uses cores - 1 as a precaution
cores <- detectCores(all.tests = FALSE, logical = TRUE)
n_cores <- cores - 1 
cl <- makePSOCKcluster(n_cores) 
registerDoParallel(cl)


## C.12 Second Tuning Race ----

# setting up the race controls
race_ctrl <- control_race(
  verbose       = TRUE, 
  verbose_elim  = TRUE,   
  save_workflow = TRUE,   
  save_pred     = TRUE, 
  randomize     = FALSE,
  burn_in       = 3,       
  parallel_over = 'everything'
)

# racing the workflows
set.seed(0304)
fine_wf_tuned_res <- fine_wf_tuned %>%
  workflow_map(
    fn        = 'tune_race_anova',
    seed      = 0304,
    resamples = five_folds,
    grid      = 300,
    metrics   = metrics,
    control   = race_ctrl
  )

# ending the parallel processing
stopCluster(cl)

#saveRDS(fine_wf_tuned_res, file = 'fine_wf_tuned_res.rds')
fine_wf_tuned_res <- readRDS('~fine_wf_tuned_res.rds')

## C.13 Best of the best Models Ranked ----

fine_tuned_rank <- rank_results(
  fine_wf_tuned_res,
  rank_metric = 'mn_log_loss',
  select_best = TRUE
)

fine_cv_rank <- fine_tuned_rank %>% 
  filter(.metric == 'mn_log_loss') %>%
  arrange(mean) %>%
  mutate(rank = row_number()) %>%
  select(rank, wflow_id, model, preprocessor, mean, std_err)

model_metric_results <- fine_cv_rank %>% 
  select(rank, wflow_id, model, mean, std_err)


## C.14 Results + Final Predictions + Visualisations ----

# best glmnet model based on log loss
best_result <- fine_wf_tuned_res %>%
  extract_workflow_set_result('glmnet_glmnet') %>%
  select_best(metric = 'mn_log_loss')

# best parameters of the best model
final_wf <- fine_wf_tuned_res %>%
  extract_workflow('glmnet_glmnet') %>%
  finalize_workflow(best_result)

final_fit <- final_wf %>%
  last_fit(split = split,
           metrics = metrics)

# Performance on test set
collect_metrics(final_fit)

# Predictions on test data
final_preds <- collect_predictions(final_fit)

all_predictions_fit <- extract_workflow(final_fit)

# predicitons on full data
complete_predictions_df <- shots_under_construction %>%
  mutate(predicted_xg = predict(all_predictions_fit , 
                                new_data = shots_under_construction, 
                                type = 'prob')$.pred_1)

# coefficient plot
# influence of each coefficient and their sign
# abs used to ignore sign for plotting bars, focusing on the size of influence instead
# sign direction comes from the colour categories
# remove intercept column
final_fit %>%
  extract_fit_parsnip() %>%
  tidy() %>%
  filter(term != '(Intercept)') %>%
  mutate(
    sign = if_else(estimate > 0, 'Positive', 'Negative'),
    term = reorder(term, abs(estimate))
  ) %>%
  ggplot(aes(x = abs(estimate), y = term, fill = sign)) +
  geom_col() +
  scale_fill_manual(values = c('Positive' = 'green', 'Negative' = 'red')) +
  labs(
    x = 'Importance',
    y = NULL,
    fill = 'Sign',
    title = 'Importance of Predictors on xG'
  ) +
  theme_minimal()

# heat map of predicted xG values
# should see yellow/green/blue clustered around goal
complete_predictions_df %>%
  arrange(predicted_xg) %>%
  ggplot(aes(x = shot_x, y = shot_y, colour = predicted_xg)) +
  annotate_pitch(dimensions = pitch_international) +
  geom_point(alpha = 0.6, size = 1.5) +
  scale_colour_viridis_c(name = 'xG') +
  theme_pitch() +
  labs(title = 'Predicted xG by Shot Location')

# player shots + total xG
# xG diff to discuss in results section
# provides player name for discussion in report
player_xg <- complete_predictions_df %>%
  group_by(wsPlayerName) %>%
  summarise(
    shots = n(),
    xg_total = sum(predicted_xg),
    goals = sum(as.numeric(as.character(goal))),
    xg_diff = goals - xg_total
  ) %>%
  filter(shots >= 20)

# scatter plot of top 10 xG overperformers
xg_over_performance_plot <- player_xg %>%
  arrange(desc(xg_diff)) %>% 
  head(10) %>%
  ggplot(aes(x = xg_total, y = goals, label = wsPlayerName)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = 'dashed') +
  geom_text_repel(size = 3) +
  labs(
    title = 'Player xG Overperformance',
    x = 'Expected Goals (xG)',
    y = 'Actual Goals'
  ) +
  theme_minimal()

# scatter plot of top 10 xG underperformers
xg_under_performance_plot <- player_xg %>%
  arrange(desc(xg_diff)) %>% 
  tail(10) %>%
  ggplot(aes(x = xg_total, y = goals, label = wsPlayerName)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = 'dashed') +
  geom_text_repel(size = 3) +
  labs(
    title = 'Player xG Underperformance',
    x = 'Expected Goals (xG)',
    y = 'Actual Goals'
  ) +
  theme_minimal()

# table showing top 5 and bottom 5 xG performers
# function to round across all columns 
player_xg_performance <- bind_rows(
  player_xg %>% slice_max(xg_diff, n = 5),
  player_xg %>% slice_min(xg_diff, n = 5)
) %>%
  mutate(across(where(is.numeric), function(x) round(x, 2)))
