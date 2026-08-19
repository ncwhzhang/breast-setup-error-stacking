# Patient-specific setup error threshold prediction using stacking ensemble meta-learning
# in breast-conserving hypofractionated radiotherapy
#
# Authors: Weiheng Zhang, Juanjuan Chen (corresponding author)
# Affiliation: Department of Oncology, the Second Affiliated Hospital of Nanchang University,
#              Nanchang 330006, Jiangxi, China
# License: MIT
#
# NOTE: The analysis dataset is NOT included in this repository (institutional data,
# not publicly shareable). Place your own de-identified data file under data/ and
# adjust variable names as needed.
#options(repos = c(CRAN="https://mirrors.tuna.tsinghua.edu.cn/CRAN"))
options(digits = 7) # 设置显示的数字精度为7位
options(show.signif.stars = FALSE) # 禁用显示 stars
options(scipen = 999) # 禁用科学计数法
options(pillar.sigfig = 7) # 控制tibble显示的有效数字为7位
options(tidymodels.dark = TRUE) # 启用 tidymodels 的暗色模式
library(tidyverse)
library(tidymodels)
library(readxl)
library(janitor)
library(future)
library(bonsai)
library(finetune)
tidymodels_prefer()
library(here)
library(baguette)
library(stacks)
library(DALEXtra)
#数据处理
rawdata <- read_excel(here("data", "SGRT_Doseerror_50.xlsx")) |>
  rename(
    "ID" = "...1",
    "axes" = "...2"
  ) |>
  fill(c(ID, CI, GI)) |>
  select(c(1:18)) |>
  pivot_longer(
    cols = 3:13,
    names_to = "setup_error",
    values_to = "dose_error"
  ) |>
  mutate(
    setup_error = as.numeric(setup_error),
    CI = round(as.numeric(CI), 3),
    action = case_when(
      dose_error < 95 ~ 1,
      dose_error >= 95 ~ 0
    ),
    ID = factor(ID),
    axes = factor(axes),
    action = factor(action, levels = c(1, 0))
  ) |>
  select(-c(dose_error))
table(rawdata$action)
#数据分割
set.seed(1501)
rawdata_split <- rawdata |>
  group_initial_split(prop = 0.8, group = ID)
rawdata_train <- rawdata_split |>
  training()
rawdata_test <- rawdata_split |>
  testing()
rawdata_train |>
  distinct(ID)
rawdata_test |>
  distinct(ID)
# 结局事件数（D95 < 95%）：总体 1650 个观测、训练集 1320、测试集 330
rawdata |> 
  filter(action == 1)
#573
rawdata_train |> 
  filter(action == 1)
#455
rawdata_test |> 
  filter(action == 1)
#118
# 重复5次10折交叉验证
set.seed(1502)
rawdata_folds <- rawdata_train |>
  group_vfold_cv(v = 5, repeats = 5, group = ID)
#-------------------------------------------------
#描述性统计
#-------------------------------------------------
# 按患者去重（假设 ID 是患者标识）
patient_level <- rawdata %>% 
  distinct(ID, .keep_all = TRUE)  # 或用 group_by(ID) %>% slice(1)

# 计算描述统计
patient_level %>% 
  summarise(
    n = n(),
    mean_age = mean(Age),
    sd_age = sd(Age),
    median_age = median(Age),
    q1 = quantile(Age, 0.25),
    q3 = quantile(Age, 0.75),
    min_age = min(Age),
    max_age = max(Age)
  )
#----------------------------------------------
#创建模型工作流
#----------------------------------------------
#--------------
#树类工作流四种
#--------------
#树模型数据预处理配方
tree_rec <- recipe(action ~ ., data = rawdata_train) |>
  update_role(ID, new_role = "ID") |>
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_numeric_predictors())
#决策树
decision_tree_rpart_spec <-
  decision_tree(
    tree_depth = tune(),
    min_n = tune(),
    cost_complexity = tune()
  ) |>
  set_engine('rpart') |>
  set_mode('classification')
# 条件推断树模型规范
ctree_spec <-
  decision_tree(
    tree_depth = tune(),
    min_n = tune()
  ) |>
  set_engine(
    "partykit",
    conditional = TRUE
  ) |>
  set_mode("classification")
#随机森林模型规范
rand_forest_randomForest_spec <-
  rand_forest(mtry = tune(), min_n = tune(), trees = 1000) |>
  set_engine('randomForest') |>
  set_mode('classification')
#装袋决策树
bag_tree_C5.0_spec <-
  bag_tree() |>
  set_engine('C5.0') |>
  set_mode("classification")
#创建树模型的工作流集合
tree_wflow <- workflow_set(
  preproc = list(tree_rec),
  models = list(
    DT = decision_tree_rpart_spec,
    CT = ctree_spec,
    RF = rand_forest_randomForest_spec,
    BAGTREE = bag_tree_C5.0_spec
  )
)
#--------------
#逻辑回归工作流1种
#--------------
#逻辑回归模型数据预处理配方
linear_rec <- recipe(action ~ ., data = rawdata_train) |>
  update_role(ID, new_role = "ID") |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors()) |>
  step_poly(all_numeric_predictors()) |>
  step_interact(~ all_numeric_predictors():all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors())
#逻辑回归模型规范
logistic_reg_glmnet_spec <-
  logistic_reg(penalty = tune(), mixture = tune()) |>
  set_engine('glmnet') |>
  set_mode("classification")
#创建逻辑回归工作流
linear_wflow <- workflow_set(
  preproc = list(linear_rec),
  models = list(
    LR = logistic_reg_glmnet_spec
  )
)
#--------------
#距离梯度工作流3种
#--------------
#距离梯度模型数据预处理配方
distance_rec <- recipe(action ~ ., data = rawdata_train) |>
  update_role(ID, new_role = "ID") |>
  step_zv(all_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_YeoJohnson(all_numeric_predictors()) |>
  step_normalize(all_numeric_predictors()) |>
  step_pca(all_numeric_predictors(), num_comp = tune()) |>
  step_normalize(all_numeric_predictors()) |>
  step_corr(all_numeric_predictors(), threshold = tune("dist_corr_thresh"))
#支持向量机模型规范
svm_poly_kernlab_spec <-
  svm_poly(
    cost = tune(),
    degree = tune(),
    scale_factor = tune(),
    margin = tune()
  ) |>
  set_engine('kernlab') |>
  set_mode('classification')
#多层感知机
mlp_nnet_spec <-
  mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) |>
  set_engine('nnet') |>
  set_mode('classification')
#k近邻
nearest_neighbor_kknn_spec <-
  nearest_neighbor(
    neighbors = tune(),
    weight_func = tune(),
    dist_power = tune()
  ) |>
  set_engine('kknn') |>
  set_mode('classification')
#创建距离模型工作流集合
distance_wflow <- workflow_set(
  preproc = list(distance_rec),
  models = list(
    SVM = svm_poly_kernlab_spec,
    MLP = mlp_nnet_spec,
    KNN = nearest_neighbor_kknn_spec
  )
)
#--------------
#拟合八种模型
#--------------
#创建所有模型工作流集合
all_workflows <-
  bind_rows(tree_wflow, linear_wflow, distance_wflow) |>
  mutate(wflow_id = gsub("(recipe_)|(normalized_)", "", wflow_id))
# 为 MLP 自定义 hidden_units 的搜索范围
mlp_param <- all_workflows |>
  extract_parameter_set_dials("MLP") |>
  update(hidden_units = hidden_units(c(1, 27)))
all_workflows <- all_workflows |>
  option_add(param_info = mlp_param, id = "MLP")
all_workflows
all_workflows |>
  extract_parameter_set_dials("MLP") |>
  extract_parameter_dials("hidden_units")
all_workflows
#----------------------------------------------
#竞速调参
#----------------------------------------------
# plan(multisession, workers = 10)
# #多模型筛选竞速法
# race_ctrl <- control_race(
#   save_pred = TRUE,
#   save_workflow = TRUE,
#   parallel_over = "resamples"
# )
# group_race_results <-
#   all_workflows |>
#   workflow_map(
#     "tune_race_anova",
#     seed = 1503,
#     resamples = rawdata_folds,
#     grid = 80,
#     control = race_ctrl,
#     metrics = metric_set(roc_auc, accuracy, brier_class)
#   )
# plan(sequential)
# write_rds(
#   group_race_results,
#   here("data", "group_race_results.rds")
# )
#----------------------------------------------
#data analysis
#----------------------------------------------
#---------
# 2.3	候选模型竞速调参与性能排名
#---------
group_race_results <- read_rds(here("data", "group_race_results.rds"))
group_race_results |>
  autoplot(
    metric = "roc_auc",
    select_best = TRUE,
    type = "wflow_id"
  ) +
  theme_classic() +
  geom_text(aes(y = mean - 1 / 30, label = wflow_id), angle = 0, hjust = 0.7) +
  labs(
    y = "ROC-AUC",
    color = NULL
  ) +
  scale_color_discrete(
    labels = c(
      "DT" = "Decision Tree (DT)",
      "CT" = "Conditional Tree (CT)",
      "RF" = "Random Forest (RF)",
      "BAGTREE" = "Bootstrap Aggregating Decision Tree (BAGTREE)",
      "LR" = "Logistic Regression (LR)",
      "SVM" = "Support Vector Machine (SVM)",
      "MLP" = "MultiLayer Perceptron (MLP)",
      "KNN" = "K-Nearest Neighbors (KNN)"
    )
  ) +
  theme(legend.position = "bottom")
group_race_results |>
  rank_results(select_best = TRUE) |>
  filter(.metric == "roc_auc") |>
  print(n = Inf)
group_race_results |>
  rank_results(select_best = TRUE) |>
  filter(.metric == "accuracy")
group_race_results |>
  rank_results(select_best = TRUE) |>
  filter(.metric == "brier_class")
# 提取所有模型的最优参数
best_params_all <- group_race_results %>%
  rank_results(select_best = TRUE) %>%
  filter(.metric == "roc_auc") %>%
  select(wflow_id, .metric, mean, std_err, n, model, preprocessor) %>%
  arrange(desc(mean))
print(best_params_all, n = Inf)
# 提取每个模型的最优超参数配置
for (wflow_id in best_params_all$wflow_id) {
  cat("\n==========", wflow_id, "==========\n")
  best <- group_race_results %>%
    extract_workflow_set_result(wflow_id) %>%
    select_best(metric = "roc_auc")
  print(best)
}
#----------------------------------------------
#model stack
#----------------------------------------------
dose_error_stack <- stacks() |>
  add_candidates(group_race_results)
dose_error_stack
set.seed(1504)
ens <- blend_predictions(
  dose_error_stack,
  penalty = 10^seq(-2, -0.5, length = 20),
  metrics = metric_set(roc_auc)
)
autoplot(ens)
autoplot(ens, "weights") + 
  theme_classic() +
  scale_fill_discrete(
    labels = c(
      "decision_tree" = "Decision Tree (DT)",
      "logistic_reg" = "Logistic Regression (LR)",
      "nearest_neighbor" = "K-Nearest Neighbors (KNN)",
      "rand_forest" = "Random Forest (RF)"
    )
  ) +
  theme(legend.position = "bottom") +
  labs(
    y = "Candidate Models",
    fill = NULL
  )
ens
ens <- fit_members(ens)
ens
ens_test_pred <-
  rawdata_test |>
  bind_cols(
    predict(ens, new_data = rawdata_test, type = "class"),
    predict(ens, new_data = rawdata_test, type = "prob")
  )
ens_test_pred
ens_test_pred |>
  accuracy(action, .pred_class)
ens_test_pred |>
  roc_auc(action, .pred_1)
ens_test_pred |>
  brier_class(action, .pred_1)
ens_test_pred |> 
  roc_curve(action, .pred_1) |> 
  autoplot()
#----------------------------------------------
# 测试集指标 95% 置信区间（患者级别 bootstrap）
#----------------------------------------------
# 注意：测试集按 ID（患者）分组，同一患者内多条记录相关，
#       因此 bootstrap 在患者级别重采样，而非逐行重采样。
compute_test_ci <- function(data, id_col = "ID", n_boot = 2000, seed = 1505) {
  set.seed(seed)

  ids <- data[[id_col]]
  unique_ids <- unique(ids)
  n_ids <- length(unique_ids)

  metric_fn <- function(indices) {
    sampled_ids <- unique_ids[indices]
    boot_data <- data |>
      filter(!!sym(id_col) %in% sampled_ids)

    # AUC/Brier 需要测试样本中同时存在两类
    if (length(unique(boot_data$action)) < 2) {
      return(c(auc = NA_real_, accuracy = NA_real_, brier = NA_real_))
    }

    auc <- boot_data |>
      roc_auc(action, .pred_1) |>
      pull(.estimate)

    acc <- boot_data |>
      accuracy(action, .pred_class) |>
      pull(.estimate)

    brier <- boot_data |>
      brier_class(action, .pred_1) |>
      pull(.estimate)

    c(auc = auc, accuracy = acc, brier = brier)
  }

  boot_results <- replicate(n_boot, {
    indices <- sample(seq_len(n_ids), n_ids, replace = TRUE)
    metric_fn(indices)
  })

  # percentile 法估计 95% CI
  ci <- apply(boot_results, 1, \(x) quantile(x, c(0.025, 0.975), na.rm = TRUE))

  tibble(
    metric = c("AUC", "Accuracy", "Brier score"),
    estimate = rowMeans(boot_results, na.rm = TRUE),
    ci_lower = ci[1, ],
    ci_upper = ci[2, ]
  )
}

# Stack 测试集 CI
stack_ci <- compute_test_ci(ens_test_pred, id_col = "ID")
print(stack_ci)

#----------------------------------------------
#最优模型测试集表现确实欠佳
#----------------------------------------------
best_results <- group_race_results |>
  extract_workflow_set_result("RF") |>
  select_best(metric = "roc_auc")
best_results
rf_final_fit <- group_race_results |>
  extract_workflow("RF") |>
  finalize_workflow(best_results) |>
  last_fit(rawdata_split)
collect_metrics(rf_final_fit)

# RF 测试集 CI（与 Stack 同函数）
rf_test_pred <- rf_final_fit |>
  collect_predictions() |>
  mutate(ID = rawdata_test$ID[.row])
rf_ci <- compute_test_ci(rf_test_pred, id_col = "ID")
print(rf_ci)
collect_predictions(rf_final_fit) |> 
  roc_curve(action, .pred_1) |> 
  autoplot()
#----------------------------------------------
#随机森林和stack的roc_acu曲线
#----------------------------------------------
# Stack 的 AUC 和 ROC
stack_auc_value <- ens_test_pred |>
  roc_auc(action, .pred_1) |>
  pull(.estimate)
stack_auc_value
stack_roc <- ens_test_pred |>
  roc_curve(action, .pred_1) |>
  mutate(model = "Stack")
stack_roc
# RF 的 AUC 和 ROC
rf_preds <- collect_predictions(rf_final_fit)
rf_auc_value <- rf_preds |>
  roc_auc(action, .pred_1) |>
  pull(.estimate)
rf_roc <- rf_preds |>
  roc_curve(action, .pred_1) |>
  mutate(model = "RF")
rf_roc
# 合并
roc_combined <- bind_rows(stack_roc, rf_roc) |>
  mutate(
    model_label = case_when(
      model == "Stack" ~ paste0("Stacking Ensemble(AUC = ", round(stack_auc_value, 2), ")"),
      model == "RF"    ~ paste0("Random Forest(AUC = ", round(rf_auc_value, 2), ")")
    )
  )
roc_combined
# 画图
p_roc_combined <- ggplot(
  roc_combined,
  aes(x = 1 - specificity, y = sensitivity, color = model_label)
) +
  geom_line(linewidth = 1.2) +
  geom_abline(linetype = "dashed", color = "gray60") +
  scale_color_manual(
    name = NULL,
    values = setNames(
      c("#E41A1C", "#377EB8"),
      c(
        paste0("Stacking Ensemble(AUC = ", round(stack_auc_value, 2), ")"),
        paste0("Random Forest(AUC = ", round(rf_auc_value, 2), ")")
      )
    )
  ) +
  labs(
    x = "1 - Specificity (False Positive Rate)",
    y = "Sensitivity (True Positive Rate)",
    title = "ROC Curves: Stacking Ensemble vs Random Forest on Test Set"
  ) +
  coord_equal() +
  theme_classic() +
  theme(legend.position = "bottom")
print(p_roc_combined)
#----------------------------------------------
#模型解释
#----------------------------------------------
#----全局解释
vip_features <- names(rawdata_train)[1:8]
vip_features
vip_train <- rawdata_train |>
  select(all_of(vip_features))
explainer_stack <- explain_tidymodels(
  ens,
  data = vip_train,
  y = as.numeric(as.character(rawdata_train$action)),
  label = ""
)
set.seed(1511)
vip_stack <- explainer_stack |>
  model_parts()
plot(vip_stack) +
  labs(
    subtitle = NULL
  ) +
  theme(
    panel.grid = element_blank()
  ) +
  annotate("segment", x = -Inf, xend = Inf, y = -Inf, yend = -Inf,
           color = "black", linewidth = 0.5) +
  annotate("segment", x = -Inf, xend = -Inf, y = -Inf, yend = Inf,
           color = "black", linewidth = 0.5)
set.seed(1512)
pdp_setup_error <- explainer_stack |>
  model_profile(variables = "setup_error", N = 100)
plot(pdp_setup_error)
pdp_axes <- explainer_stack |>
  model_profile(variables = "axes", N = 100)
plot(pdp_axes)
pdp_ci <- explainer_stack |>
  model_profile(variables = "CI", N = 100)
plot(pdp_ci)
#----局部解释
test_data <- rawdata_test[3, ]
test_data
shap_test_data <-
  explainer_stack |>
  predict_parts(new_observation = test_data, type = "shap", B = 30) |>
  filter(!str_detect(variable, "^ID"))
plot(shap_test_data) +
  labs(y = NULL) +
  theme(
    plot.title = element_blank(),
    panel.grid = element_blank(),
    axis.line = element_line(color = "black")
  ) +
  annotate("segment", x = -Inf, xend = Inf, y = -Inf, yend = -Inf,
           color = "black", linewidth = 0.5) +
  annotate("segment", x = -Inf, xend = -Inf, y = -Inf, yend = Inf,
           color = "black", linewidth = 0.5)
#校准曲线
 library(probably)
ens_test_pred |>
  cal_plot_breaks(truth = action, estimate = .pred_1, num_breaks = 5)
# 用 Platt scaling / 逻辑回归校准
# estimate 需要传入两个类别的概率列（.pred_0 和 .pred_1），不能只传一列
cal_model <- cal_estimate_logistic(ens_test_pred, truth = action, estimate = c(.pred_0, .pred_1))
ens_test_pred_cal <- cal_apply(ens_test_pred, cal_model)
# 校准后的校准曲线
ens_test_pred_cal |>
  cal_plot_breaks(truth = action, estimate = .pred_1, num_breaks = 5) +
  theme_classic() +
  coord_obs_pred()
#---------------------------------------------
# 决策曲线（重要：修复 dcurves 的 event level 问题）
#---------------------------------------------
# dcurves 包把 factor 的第二个 level 当作 event（正类）。
# 你的 action 是 factor(action, levels = c("1", "0"))，所以 dcurves 把 "0" 当 event，
# 导致 DCA 完全颠倒。需要在 DCA 前把 level 改成 c("0", "1")，让 "1" 成为第二个 level。
ens_test_pred_dca <- ens_test_pred |>
  mutate(action_dca = factor(action, levels = c("0", "1")))
ens_test_pred_cal_dca <- ens_test_pred_cal |>
  mutate(action_dca = factor(action, levels = c("0", "1")))
library(dcurves)
# 未校准的 Stack DCA
dca_stack <- dca(
  action_dca ~ .pred_1,
  data = ens_test_pred_dca,
  thresholds = seq(0, 0.99, by = 0.01)
)
plot(dca_stack)
# 校准后的 Stack DCA
dca_stack_cal <- dca(
  action_dca ~ .pred_1,
  data = ens_test_pred_cal_dca,
  thresholds = seq(0, 0.99, by = 0.01)
)
plot(dca_stack_cal)
# 随机森林预测概率
rf_test_pred <- rf_final_fit |>
  collect_predictions()
# RF 也需要同样处理 event level
rf_test_pred_dca <- rf_test_pred |>
  mutate(action_dca = factor(action, levels = c("0", "1")))
dca_rf <- dca(
  action_dca ~ .pred_1,
  data = rf_test_pred_dca,
  thresholds = seq(0, 0.99, by = 0.01)
)
plot(dca_rf)
#---------------------------------------------
# DCA：全模型对比，Stack 高亮，其他淡化
#---------------------------------------------
# 1) 先确认 workflow ID 名称
group_race_results |> 
  rank_results(select_best = TRUE) |> 
  distinct(wflow_id)

# 2) 定义候选模型 ID（请根据上一步输出核对）
candidate_ids <- c("DT", "CT", "RF", "BAGTREE", "LR", "SVM", "MLP", "KNN")
dca_all_data <- rawdata_test |>
  select(action) |>
  mutate(
    action_dca = factor(action, levels = c("0", "1")),
    Stack = ens_test_pred$.pred_1
  )
for (m_id in candidate_ids) {
  best_m <- group_race_results |>
    extract_workflow_set_result(m_id) |>
    select_best(metric = "roc_auc")
  
  final_fit_m <- group_race_results |>
    extract_workflow(m_id) |>
    finalize_workflow(best_m) |>
    last_fit(rawdata_split)
  
  pred_m <- final_fit_m |>
    collect_predictions() |>
    select(.pred_1)
  
  dca_all_data[[m_id]] <- pred_m$.pred_1
}
dca_formula <- as.formula(
  paste("action_dca ~ Stack +", paste(candidate_ids, collapse = " + "))
)
dca_all <- dca(
  dca_formula,
  data = dca_all_data,
  thresholds = seq(0, 0.99, by = 0.01)
)
# 画图：Stack 突出，其他淡化
p_dca_highlight <- dca_all$dca |>
  mutate(
    legend_group = case_when(
      variable == "Stack" ~ "Stacking Ensemble",
      variable == "all" ~ "All Positive",
      variable == "none" ~ "All Negative",
      TRUE ~ "All Candidate Models"
    ),
    legend_group = factor(
      legend_group,
      levels = c(
        "Stacking Ensemble",
        "All Candidate Models",
        "All Positive",
        "All Negative"
      )
    )
  ) |>
  ggplot(aes(x = threshold, y = net_benefit, group = variable)) +
  geom_line(
    aes(
      color = legend_group,
      linetype = legend_group,
      linewidth = legend_group,
      alpha = legend_group
    )
  ) +
  scale_color_manual(
    name = NULL,
    values = c(
      "Stacking Ensemble" = "#E41A1C",
      "All Candidate Models" = "gray75",
      "All Positive" = "black",
      "All Negative" = "black"
    )
  ) +
  scale_linetype_manual(
    name = NULL,
    values = c(
      "Stacking Ensemble" = "solid",
      "All Candidate Models" = "solid",
      "All Positive" = "solid",
      "All Negative" = "dashed"
    )
  ) +
  scale_linewidth_manual(
    name = NULL,
    values = c(
      "Stacking Ensemble" = 1.5,
      "All Candidate Models" = 0.6,
      "All Positive" = 0.8,
      "All Negative" = 0.8
    )
  ) +
  scale_alpha_manual(
    name = NULL,
    values = c(
      "Stacking Ensemble" = 1,
      "All Candidate Models" = 0.7,
      "All Positive" = 1,
      "All Negative" = 1
    )
  ) +
  labs(
    x = "Threshold Probability",
    y = "Net Benefit",
    title = "Decision Curve Analysis: Stacking Ensemble(red) vs All Candidate Models (gray)"
  ) +
  coord_cartesian(ylim = c(-0.05, 0.4)) +
  theme_classic() +
  theme(
    legend.position = "bottom",
    legend.key.width = unit(1.2, "cm")
  )
print(p_dca_highlight)
#分面图
p_dca_facet <- ggplot(
  dca_all$dca |> filter(!variable %in% c("all", "none")),
  aes(x = threshold, y = net_benefit, color = variable)
) +
  geom_line(linewidth = 1) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  facet_wrap(~ variable, ncol = 3) +
  labs(
    x = "Threshold Probability",
    y = "Net Benefit",
    title = "Decision Curve Analysis by Model"
  ) +
  coord_cartesian(ylim = c(-0.1, 0.5)) +
  theme_minimal() +
  theme(legend.position = "none")
print(p_dca_facet)
#----------------------
#病人个体化预测
#----------------------
# 1) 定义病人的固定特征（除了 setup_error）
# ILLUSTRATIVE EXAMPLE ONLY -- the values below are synthetic and do NOT
# correspond to any study participant. Replace with your own patient's values.
patient_id <- "DEMO01"
patient_axis <- "Z"  # 可以是 "X"、"Y" 或 "Z"
patient_CI <- 1.05
patient_GI <- 1.80
patient_BMI <- 24.0
patient_volume_ptv <- 400   # 根据该病人实际值填写
patient_Age <- 50           # 根据该病人实际值填写
# 2) 构建 setup_error 扫描序列
setup_error_seq <- seq(-0.5, 0.5, by = 0.001)

# 3) 生成预测数据框
# 注意：因子变量的 levels 必须与训练集一致
patient_scan <- tibble(
  ID = factor(rep(patient_id, length(setup_error_seq)), levels = levels(rawdata_train$ID)),
  axes = factor(rep(patient_axis, length(setup_error_seq)), levels = levels(rawdata_train$axes)),
  setup_error = setup_error_seq,
  CI = rep(patient_CI, length(setup_error_seq)),
  GI = rep(patient_GI, length(setup_error_seq)),
  BMI = rep(patient_BMI, length(setup_error_seq)),
  volume_ptv = rep(patient_volume_ptv, length(setup_error_seq)),  # 新增
  Age = rep(patient_Age, length(setup_error_seq))                  # 新增
)
# 4) 用 stacking 元学习模型预测概率
patient_pred <- patient_scan |>
  bind_cols(
    predict(ens, new_data = patient_scan, type = "prob")
  )

# 5) 找出预测概率最接近 0.5 的 setup_error（默认决策阈值）
threshold_05 <- patient_pred |>
  mutate(distance = abs(.pred_1 - 0.5)) |>
  slice_min(distance, n = 1)

threshold_05

# 6) 找出 action 从 0 跳变到 1 的转变点（更直观）
change_points <- patient_pred |>
  mutate(pred_class = ifelse(.pred_1 > 0.5, 1, 0)) |>
  filter(pred_class - dplyr::lag(pred_class) != 0)

change_points

# 7) 可视化
ggplot(patient_pred, aes(x = setup_error, y = .pred_1)) +
  geom_line(color = "#377EB8", linewidth = 1.2) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "red") +
  geom_point(
    data = change_points,
    aes(x = setup_error, y = .pred_1),
    color = "red", size = 3
  ) +
  geom_vline(
    data = change_points,
    aes(xintercept = setup_error),
    linetype = "dashed", color = "red"
  ) +
  geom_text(
    data = change_points,
    aes(
      x = setup_error,
      y = .pred_1,
      label = paste0("Threshold = ", round(setup_error, 2), " cm")
    ),
    color = "red", vjust = -1, hjust = 0
  ) +
  labs(
    x = "Setup error (cm)",
    y = "Probability of intervention\n(D95 < 95%)",
    title = paste0(
      "Patient-specific setup error threshold prediction\n",
      "Axis = ", patient_axis,
      ", CI = ", patient_CI, ", GI = ", patient_GI, ", BMI = ", patient_BMI,
      ", PTV Volume = ", patient_volume_ptv, ", Age = ", patient_Age
    )
  ) +
  scale_y_continuous(limits = c(-0.05, 1.05), breaks = seq(0, 1, by = 0.2)) +
  theme_classic()
