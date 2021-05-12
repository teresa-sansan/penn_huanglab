library('pROC')
overall <- read.table(file = 'validation_testing_comparison.tsv', sep = '\t', header = TRUE)
domi <- read.table(file = 'validation_testing_domi_comparison.tsv', sep = '\t', header = TRUE)
recess <- read.table(file = 'validation_testing_recess_comparison.tsv', sep = '\t', header = TRUE)


delong <- function(data, a, b){
  roc_data1 <- roc(data[,1], data[,a])
  roc_data2 <- roc(data[,1], data[,b])
  result <- roc.test(roc_data1, roc_data2,  method = c('delong'))
  return(result$p.value)
}

log_gradient <- delong(overall, a = 2, b = 4)
print('logistic vs. gradient boosting=')
print(log_gradient)
print('')

log_svm <- delong(overall, a = 2, b = 3)
print('logistic vs. svm = ')
print(log_svm)
print('')

gradient_svm <- delong(overall, a = 3, b = 4)
print('svm vs. gradient boosting')
print(gradient_svm)
print('')

print(delong(domi, 2,4))
print(delong(recess, 2,4))
