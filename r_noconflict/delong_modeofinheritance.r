library('pROC')
data <- read.table(file = 'omit_remove_partial.tsv', sep = '\t', header = TRUE)
delong <- function(data, a, b ){
  roc_data1 <- roc(data[,1], data[,a])
  roc_data2 <- roc(data[,1], data[,b])
  result <- roc.test(roc_data1, roc_data2,  method = c('delong'))
  print(result)
  return(result$p.value)
}


domivsrecess <- delong(data, a = 2, b = 3)
print('domi vs. recess =')
print(domivsrecess)
print('')

#domivsomit <- delong(data, a = 2, b = 4)
#print('domi vs.omit = ')
#print(domivsomit)
#print('')

#recessvsomit <- delong(data, a=3, b=4)
#print('recess vs omit =')
#print(recessvsomit)



