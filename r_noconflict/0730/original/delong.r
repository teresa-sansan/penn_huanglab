library('pROC')
data <- read.table(file = 'overall_comparison.tsv', sep = '\t', header = TRUE)
delong <- function(data1, data2, a = 2, b = 2){
  roc_data1 <- roc(data1[,1], data1[,a], direction = '>')
  roc_data2 <- roc(data2[,1], data2[,b], direction = '>')
  result <- roc.test(roc_data1, roc_data2,  method = c('delong'))
  return(result$p.value)
}

log_gradient <- delong(data,data, a = 2, b = 4)
print('logistic vs. gradient boosting')
print(log_gradient)
print('')

log_svm <- delong(data, data, a = 2, b = 3)
print('logistic vs. svm  ')
print(log_svm )
print('')

gradient_svm <- delong(data, data, a = 3, b = 4)
print('svm vs. gradient boosting')
print(gradient_svm)
print('')

## only for dominant and recessive data
#gb vs opposite model in original model
opposite_gb <- delong(data, data, a = 4, b = 5)

#print('gradiant boosting in one vs gradient boosting from other model')
#print(opposite_gb )
#print('')

##gb vs all gb
#opposite_omit_gb <- delong(data, data, a = 4, b = 6)
#print('gradiant goosting in gb vs from All')
#print(opposite_omit_gb)

