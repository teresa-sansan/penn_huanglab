library('pROC')
overall <- read.table(file = 'overall_comparison.tsv', sep = '\t', header = TRUE)
delong <- function(data1, data2, a = 2, b = 2){
  roc_data1 <- roc(data1[,1], data1[,a])
  roc_data2 <- roc(data2[,1], data2[,b])
  result <- roc.test(roc_data1, roc_data2,  method = c('delong'))
  return(result$p.value)
}

log_gradient <- delong(overall,overall, a = 2, b = 4)
print('logistic vs. gradient boosting=')
print(log_gradient)
print('')

log_svm <- delong(overall, overall, a = 2, b = 3)
print('logistic vs. svm = ')
print(log_svm)
print('')

gradient_svm <- delong(overall, overall, a = 3, b = 4)
print('svm vs. gradient boosting')
print(gradient_svm)
print('')
