import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import*
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def preprocess (file, col_name, Balanced = False):
    if("." in file):
        try:
            file = pd.read_csv(file, sep = '\t')
            #print('successfully open file')
        except IOError:
            print("couldn't find file {}".format(file))
    col = file.loc[:,[col_name]]
    total_len = col.shape[0]
    summary = np.sum(col)[0]
    if total_len/2 == summary:
        print('Balanced,')
        balanced = 1
    else:
        print('Unbalanced,')
        balanced = 0
        print('with num of 0 is {}, num of 1 is {}.\n'.format(total_len-summary, summary))

    zero = file.loc[col[col_name]==0]
    one = file.loc[col[col_name]==1]
#start balancing
    if Balanced == True and balanced == 0:
        print('start balancing...')
        if(total_len/2 > summary): # 0>1
            zero = zero.sample(n = summary, random_state = 42)
        else: # 1>0
            one = one.sample(n = total_len - summary, random_state = 42)
        print('with num of 0 is {}, num of 1 is {}.\n'.format(min(summary,total_len-summary), min(summary,total_len-summary)))
    output = pd.concat([zero,one])
    return(output)

def splitNfit(dataframe, Xname, Yname, regression = 0,testsize = 0.3, regulation = False):
    y = dataframe.loc[:,[Yname]].values
    y = y.flatten()
    X = dataframe.loc[:,Xname]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = testsize, random_state = 42)
   
    if regression == 0:
        if regulation == False :
            regression = LogisticRegression(random_state = 42 , solver = 'lbfgs').fit(X_train, y_train)
        else:
            regression = LogisticRegression(random_state = 42 , penalty = regulation).fit(X_train, y_train)
    test_hat = regression.predict_proba(X_test)[:,1]
    return(X_train, X_test, y_train, y_test, test_hat, regression)
    
  
def drawROC(ytest, ytest_hat, lw=3, linestyle = '--', label = '', lastone = False, MoreThanOnelocation = 'lower right', fontsize = 12, title = ''):
    fpr, tpr, thresholds = roc_curve(ytest, ytest_hat)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr, tpr, lw = lw, linestyle = linestyle, label = label + ', (AUC = %0.3f)'%roc_auc)
    if lastone == True:
        plt.legend(loc = 'lower right', fontsize = 12)
        plt.title(title)
        plt.show()

        