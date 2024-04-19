
require(data.table)
library(tidyverse)

h2 <- fread("data/h2_hourly_gb.csv")
h2[,targetTime:=as.POSIXct(arrival_1h,tz="UTC",format="%Y-%m-%dT%H:%M:%SZ")]
h2[,targetTime_UK:=targetTime]; attributes(h2$targetTime_UK)$tzone <- "Europe/London"
# write.csv(h2, 'results_to_python/observations.csv', row.names = FALSE)
as_tibble(h2) %>% write_csv('results_to_python/observations.csv')

load("results/JethroResults_pt1_2021-10-08.Rda")
load("results/JethroResults_pt2_2021-10-08.Rda")
JB_results <- c(temp1,temp2); rm(temp1,temp2)

tbats <- data.table(readRDS("results/tbats_bahman.rds"))
setnames(tbats,old = c("origin","target","point_forecast"),c("issueTime","targetTime_UK","expectation"))
tbats[,issueTime := issueTime+3600]

# Faster is missing
# faster <- data.table(readRDS("results/fasster_bahman.rds"))
# setnames(faster,old = c("point_forecast"),c("expectation"))
# faster[,issueTime := issueTime+3600]

prophet <- data.table(readRDS("results/prophet_bahman.rds"))
setnames(prophet,old = c("point_forecast"),c("expectation"))
prophet[,issueTime := issueTime+3600]

load("results/IvanValues.RData")


# For lists:
# iterate over names
# Compute Dataframe with issueTime, targetTime_UK, expectation
# Export to csv
save_df <- function (df_fc, name) {
  df_subset <- as_tibble(df_fc[ , c('issueTime', 'targetTime_UK', 'expectation')])
  df_subset <- df_subset %>% filter(issueTime >= "2018-03-01")
  write_csv(df_subset, paste0('results_to_python/', name, '.csv'))
}


for (n in names(JB_results)) {
  save_df(JB_results[[n]], n)
}
save_df(prophet, 'prophet')
save_df(tbats, 'tbats')
for (n in names(quantileValuesIvan)) {
  df_fc  <- quantileValuesIvan[[n]]
  df_fc$targetTime_UK <- force_tz(df_fc$targetTime_UK, "UTC")
  df_fc$issueTime <- force_tz(df_fc$issueTime, "UTC")
  save_df(df_fc, n)
}

# tmp <- as_tibble(JB_results[['Benchmark_1']])
# tmp %>% mutate(h = hour(issueTime)) %>% count(h)



# Include quantiles -------------------------------------------------------


h2 <- fread("data/h2_hourly_gb.csv")
h2[,targetTime:=as.POSIXct(arrival_1h,tz="UTC",format="%Y-%m-%dT%H:%M:%SZ")]
h2[,targetTime_UK:=targetTime]; attributes(h2$targetTime_UK)$tzone <- "Europe/London"
as_tibble(h2) %>% write_csv('results_to_python_quantiles/observations.csv')

load("results/JethroResults_pt1_2021-10-08.Rda")
load("results/JethroResults_pt2_2021-10-08.Rda")
JB_results <- c(temp1,temp2); rm(temp1,temp2)

tbats <- data.table(readRDS("results/tbats_bahman.rds"))
setnames(tbats,old = c("origin","target","point_forecast"),c("issueTime","targetTime_UK","expectation"))
tbats[,issueTime := issueTime+3600]

# Faster is missing
# faster <- data.table(readRDS("results/fasster_bahman.rds"))
# setnames(faster,old = c("point_forecast"),c("expectation"))
# faster[,issueTime := issueTime+3600]

prophet <- data.table(readRDS("results/prophet_bahman.rds"))
setnames(prophet,old = c("point_forecast"),c("expectation"))
prophet[,issueTime := issueTime+3600]

load("results/IvanValues.RData")


# For lists:
# iterate over names
# Compute Dataframe with issueTime, targetTime_UK, expectation
# Export to csv
save_df <- function (df_fc, name) {
  df_subset <- as_tibble(df_fc)
  df_subset <- df_subset %>% filter(issueTime >= "2018-03-01")
  write_csv(df_subset, paste0('results_to_python_quantiles/', name, '.csv'))
}


for (n in names(JB_results)) {
  save_df(JB_results[[n]], n)
}
save_df(prophet, 'prophet')
save_df(tbats, 'tbats')
for (n in names(quantileValuesIvan)) {
  df_fc  <- quantileValuesIvan[[n]]
  df_fc$targetTime_UK <- force_tz(df_fc$targetTime_UK, "UTC")
  df_fc$issueTime <- force_tz(df_fc$issueTime, "UTC")
  save_df(df_fc, n)
}

