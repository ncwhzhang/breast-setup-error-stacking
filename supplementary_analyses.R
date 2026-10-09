# ============================================================
# Supplementary robustness and sensitivity analyses (revision R1, FINAL)
# Run AFTER the main script TM_Dose_error_group_analysis.R
# Required objects in the environment: rawdata, rawdata_train, rawdata_test,
#                                      group_race_results, ens
# All results are saved to revision_output/ (RDS data + PNG figures)
# Expected total runtime: several hours (160 stack builds); run overnight
# ============================================================
library(tidyverse)
library(tidymodels)
library(stacks)
library(readxl)

set.seed(2026)
dir.create("revision_output", showWarnings = FALSE)

# Path to the raw data file (same as in the main script; replace with a full
# path string if here() fails)
xlsx_path <- here::here("stats_learning", "r_learning", "SML_TMwR", "SGRT_Doseerror_50.xlsx")

# ------------------------------------------------------------
# 0. Helper: rebuild the stack from the best hyperparameters of each model
#    in the racing results (identical members and meta-learner as main analysis)
# ------------------------------------------------------------
build_stack <- function(train_data, folds, seed = 1503) {
  best_list <- list()
  for (m_id in c("DT", "CT", "RF", "BAGTREE", "LR", "SVM", "MLP", "KNN")) {
    best_list[[m_id]] <- group_race_results |>
      extract_workflow_set_result(m_id) |>
      select_best(metric = "roc_auc")
  }
  wfs <- list()
  for (m_id in names(best_list)) {
    wfs[[m_id]] <- group_race_results |>
      extract_workflow(m_id) |>
      finalize_workflow(best_list[[m_id]])
  }
  ctrl <- control_grid(save_pred = TRUE, save_workflow = TRUE,
                       parallel_over = "resamples")
  res <- purrr::map(wfs, ~ fit_resamples(.x, resamples = folds, control = ctrl,
                                         metrics = metric_set(roc_auc)))
  race_like <- do.call(as_workflow_set, res)
  st <- stacks() |> add_candidates(race_like)
  ens <- blend_predictions(st, penalty = 10^seq(-2, -0.5, length = 20),
                           metrics = metric_set(roc_auc))
  fit_members(ens)
}

# Return NA instead of aborting the whole (overnight) run if one fold fails
build_stack_safe <- function(train_data, folds, seed = 1503) {
  tryCatch(build_stack(train_data, folds, seed),
           error = function(e) { message("  build failed: ", conditionMessage(e)); NULL })
}

# Nested evaluation: outer 5-fold group CV; the complete stack is rebuilt
# inside each outer fold; returns the 5 fold AUCs
nested_cv_auc <- function(data, seed_base = 5000) {
  outer <- group_vfold_cv(data, v = 5, group = ID)
  purrr::map_dbl(seq_len(5), function(f) {
    tr2 <- analysis(outer$splits[[f]])
    va2 <- assessment(outer$splits[[f]])
    folds2 <- group_vfold_cv(tr2, v = 5, group = ID)
    ens2 <- build_stack_safe(tr2, folds2, seed = seed_base + f)
    if (is.null(ens2)) return(NA_real_)
    p <- predict(ens2, va2, type = "prob")
    roc_auc_vec(va2$action, p$.pred_1)
  })
}

# ------------------------------------------------------------
# 1. Learning curve (complete stacking pipeline, nested evaluation)
#    5 sizes x 4 repeats x 5 outer folds = 100 stack builds
# ------------------------------------------------------------
message("=== Part 1: learning curve (full stack) ===  ", Sys.time())

learning_curve_stack <- function(train_data, sizes = c(8, 16, 24, 32, 40), reps = 4) {
  ids <- unique(train_data$ID)
  out <- list()
  k <- 1
  for (n in sizes) {
    for (r in seq_len(reps)) {
      set.seed(1000 + n * 10 + r)
      sub_ids <- sample(ids, n)
      tr_sub <- train_data |> filter(ID %in% sub_ids)
      aucs <- nested_cv_auc(tr_sub, seed_base = 5000)
      out[[k]] <- tibble(n_patients = n, rep = r,
                         mean = mean(aucs, na.rm = TRUE),
                         n_folds_ok = sum(!is.na(aucs)))
      k <- k + 1
      message("n=", n, " rep=", r, " AUC=", round(mean(aucs, na.rm = TRUE), 3),
              "  ", Sys.time())
    }
  }
  bind_rows(out)
}

lc_stack <- learning_curve_stack(rawdata_train)
saveRDS(lc_stack, "revision_output/lc_stack.rds")

lc_stack |> group_by(n_patients) |>
  summarise(mean_auc = mean(mean), sd_auc = sd(mean), .groups = "drop") |> print()

p_lc <- ggplot(lc_stack, aes(x = factor(n_patients), y = mean)) +
  geom_boxplot(fill = "#BBDEFB") +
  geom_jitter(width = 0.15, alpha = 0.5) +
  labs(x = "Number of training patients", y = "Cross-validated AUC") +
  theme_classic()
ggsave("revision_output/fig7A_learning_curve_R.png", p_lc,
       width = 6, height = 4, dpi = 300)

# ------------------------------------------------------------
# 2. 20 repeated random patient-level splits (complete pipeline refitted)
# ------------------------------------------------------------
message("=== Part 2: 20 repeated splits ===  ", Sys.time())

repeated_splits <- function(data, n_splits = 20) {
  out <- vector("list", n_splits)
  for (s in seq_len(n_splits)) {
    sp <- group_initial_split(data, prop = 0.8, group = ID)
    tr <- training(sp)
    te <- testing(sp)
    folds <- group_vfold_cv(tr, v = 5, group = ID)
    ens_s <- build_stack_safe(tr, folds, seed = 1503 + s)
    if (is.null(ens_s)) { out[[s]] <- tibble(auc = NA, acc = NA, brier = NA, split = s); next }
    pred_s <- predict(ens_s, te, type = "prob") |> bind_cols(te |> select(action))
    out[[s]] <- pred_s |>
      summarise(auc = roc_auc_vec(action, .pred_1),
                acc = accuracy_vec(action, factor(ifelse(.pred_1 > 0.5, 1, 0), levels = c(1, 0))),
                brier = brier_class_vec(action, .pred_1)) |>
      mutate(split = s)
    message("split ", s, " done  ", Sys.time())
  }
  bind_rows(out)
}

rs_res <- repeated_splits(rawdata)
saveRDS(rs_res, "revision_output/rs_res.rds")

rs_res |> summarise(median_auc = median(auc), q1 = quantile(auc, .25),
                    q3 = quantile(auc, .75), min = min(auc), max = max(auc)) |> print()

p_rs <- ggplot(rs_res, aes(x = "", y = auc)) +
  geom_boxplot(fill = "#BBDEFB", width = 0.4) +
  geom_jitter(width = 0.08, alpha = 0.6) +
  labs(x = "20 repeated 80:20 patient-level splits", y = "Test AUC") +
  theme_classic()
ggsave("revision_output/fig7B_repeated_splits_R.png", p_rs,
       width = 6, height = 4, dpi = 300)

# ------------------------------------------------------------
# 3. Outcome-definition sensitivity (complete stack, nested evaluation;
#    includes the primary threshold 95) -- revision Table 4
# ------------------------------------------------------------
message("=== Part 3: threshold sensitivity (full stack) ===  ", Sys.time())

threshold_sensitivity_stack <- function(raw_xlsx_path, thresholds = c(93, 94, 95, 96, 97)) {
  raw0 <- read_excel(raw_xlsx_path) |>
    rename("ID" = "...1", "axes" = "...2") |>
    fill(c(ID, CI, GI)) |> select(c(1:18))
  out <- list()
  for (thr in thresholds) {
    d <- raw0 |>
      pivot_longer(cols = 3:13, names_to = "setup_error", values_to = "dose_error") |>
      mutate(setup_error = as.numeric(setup_error),
             action = factor(as.integer(dose_error < thr), levels = c(1, 0)),
             ID = factor(ID), axes = factor(axes)) |>
      select(-dose_error)
    tr <- d[d$ID %in% rawdata_train$ID, ]   # same patient-level split as main analysis
    aucs <- nested_cv_auc(tr, seed_base = 7000)
    out[[as.character(thr)]] <- tibble(threshold = thr,
                                       events = sum(d$action == 1),
                                       mean_auc = mean(aucs, na.rm = TRUE),
                                       sd_auc = sd(aucs, na.rm = TRUE),
                                       n_folds_ok = sum(!is.na(aucs)))
    message("threshold ", thr, " done  ", Sys.time())
  }
  bind_rows(out)
}

sens_stack <- threshold_sensitivity_stack(xlsx_path)
saveRDS(sens_stack, "revision_output/sens_stack.rds")
sens_stack |> print()

# ------------------------------------------------------------
# 4. Repeated 5-fold group CV of the complete stacking pipeline
#    (full cohort, 3 repeats)
# ------------------------------------------------------------
message("=== Part 4: repeated 5-fold CV (full cohort) ===  ", Sys.time())

repeated_cv_stack <- function(data, reps = 3) {
  out <- list()
  for (r in seq_len(reps)) {
    folds <- group_vfold_cv(data, v = 5, repeats = 1, group = ID)
    for (f in seq_len(5)) {
      tr <- analysis(folds$splits[[f]]); va <- assessment(folds$splits[[f]])
      folds_tr <- group_vfold_cv(tr, v = 5, group = ID)
      ens_f <- build_stack_safe(tr, folds_tr, seed = 2000 + 10 * r + f)
      if (is.null(ens_f)) {
        out[[paste(r, f)]] <- tibble(rep = r, fold = f, auc = NA, acc = NA, brier = NA)
        next
      }
      pred_f <- predict(ens_f, va, type = "prob") |> bind_cols(va |> select(action))
      out[[paste(r, f)]] <- pred_f |>
        summarise(rep = !!r, fold = !!f, auc = roc_auc_vec(action, .pred_1),
                  acc = accuracy_vec(action, factor(ifelse(.pred_1 > 0.5, 1, 0), levels = c(1, 0))),
                  brier = brier_class_vec(action, .pred_1))
    }
    message("repeat ", r, " done  ", Sys.time())
  }
  bind_rows(out)
}

cv_res <- repeated_cv_stack(rawdata)
saveRDS(cv_res, "revision_output/cv_res.rds")

cv_res |> summarise(mean_auc = mean(auc), sd_auc = sd(auc),
                    min_auc = min(auc), max_auc = max(auc)) |> print()

# ------------------------------------------------------------
# 5. Patient-level bootstrap 95% CI for the test-set AUC
#    (uses the fitted ensemble 'ens' from the main script; takes seconds)
# ------------------------------------------------------------
set.seed(2026)
pred_test <- predict(ens, rawdata_test, type = "prob") |>
  bind_cols(rawdata_test |> select(ID, action))
boot_auc <- replicate(2000, {
  ids_b <- sample(unique(pred_test$ID), replace = TRUE)
  d_b <- purrr::map_dfr(seq_along(ids_b), function(i) pred_test |> filter(ID == ids_b[i]))
  roc_auc_vec(d_b$action, d_b$.pred_1)
})
quantile(boot_auc, c(0.025, 0.5, 0.975)) |> print()
saveRDS(boot_auc, "revision_output/boot_auc.rds")

# ------------------------------------------------------------
# Done
# ------------------------------------------------------------
message("=== ALL DONE ===  ", Sys.time())
message("Results in revision_output/: lc_stack.rds, rs_res.rds, sens_stack.rds, cv_res.rds, boot_auc.rds + 2 PNGs")
