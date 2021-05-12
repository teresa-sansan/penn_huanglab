import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import*
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn import metrics, svm
import pylab as pl
import statsmodels.api as sm

import math

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
    
 
## ROC curve without output figure
def drawROC(ytest, ytest_hat, lw=3, linestyle = '--', label = '', lastone = False, MoreThanOnelocation = 'lower right', fontsize = 12, title = '', legendloc = False):
    fpr, tpr, thresholds = roc_curve(ytest, ytest_hat)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr, tpr, lw = lw, linestyle = linestyle, label = label + ', (AUC = %0.3f)'%roc_auc)
    if lastone == True:
        if legendloc != False:
            ax.legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=lengend,borderaxespad=0, frameon=False)
        else:
            plt.legend(loc = 'lower right', fontsize = 12)
        plt.title(title)
        plt.show()


## ROC CURVE with output figure        
def get_ROC(ax,ytest, ytest_hat, label, lastone = False, x = 'False Positive Rate', y = 'True Positive Rate', title = None, legendloc = 'lower right', lw = 2, linestyle = '-'):
    fpr, tpr, thresholds = roc_curve(ytest, ytest_hat)
    auroc = auc(fpr,tpr)
    ax.plot(fpr, tpr, lw = lw, linestyle = linestyle, label = label +', (AUC = %0.3f)'%auroc)
    if(lastone == True):
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title)
        ax.legend(loc='lower right',fontsize = 12)

        
def data_tun(data, model, tuned_parameters):
    if(data == 'overall'):
        Xtrain, Xtest, ytrain, ytest = X_train, X_test, y_train, y_test
    elif(data == 'domi'):
        Xtrain, Xtest, ytrain, ytest = X_train_domi, X_test_domi, y_train_domi, y_test_domi
    else:
        Xtrain, Xtest, ytrain, ytest = X_train_recess, X_test_recess, y_train_recess, y_test_recess
    tuning(Xtrain, Xtest, ytrain, ytest, model, tuned_parameters) 

def tuning(Xtrain, ytrain, model, tuned_parameters):
    print("# Tuning hyper-parameters for %s" % 'recall')
    print()
    
    if(model == 'svm'):
        clf = GridSearchCV(SVC(), tuned_parameters, scoring='%s_macro' % 'recall')
        
    if(model == 'gradient boosting'):
        clf = GridSearchCV(GradientBoostingClassifier(), tuned_parameters, scoring='%s_macro' % 'recall')
        
    clf.fit(Xtrain, ytrain)

    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print("With score : %0.3f\n"%clf.best_score_)
    print("Grid scores on development set:\n")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    print("Detailed classification report:\n")
    print("The model is trained on the full development set.")
    print()
    
    
def fitting_gb(Xtrain, ytrain, Xtest, ytest, learning, depth, estimator, fitmodel = False):
    if(fitmodel == False):
        model = GradientBoostingClassifier(learning_rate = learning, max_depth = depth, n_estimators = estimator, subsample = 1)
        model.fit(Xtrain, ytrain)
    else:
        model = fitmodel
    
    model_pred = model.decision_function(Xtest)
    
    return(ytest, model_pred, model)
        
        


def enrichment(df, portion, features, log = True):
    num = int(df.shape[0]*portion*0.01)
    oddsratios = []
    errors = []
    for i in features:
        
        df[i] = pd.to_numeric(df[i])
        top = np.sum(df.nlargest(num,i)['result']) 
        array1 = [top, num-top]
        #print(matrix1)
    
        #matrix = contingency_table([1]*num, df.nlargest(num,i)['result'], num)
        #matrix = confusion_matrix(abs(df.nlargest(num,i)['result']-1),[0]*num)
              
        tail = np.sum(df.nsmallest(df.shape[0]-num,i)['result'])
        array2 = [tail, df.shape[0]-num-tail]
        #print(array2)
        #print(np.asarray([array1, array2]))
        
        result = sm.stats.Table2x2(np.asarray([array1, array2])).oddsratio
    
        if(log == True):
            result = math.log2(result)
            #result = math.log(result)
        #print(result)
        oddsratios.append(result)

        # check
        #print((top/(num-top))/(tail/(df.shape[0]-num-tail)))
        

        # change of base
        error = math.sqrt(1/top + 1/(num-top) + 1/tail + 1/(df.shape[0]-num-tail))* math.log2(math.e)
        #error = math.sqrt(1/top + 1/(num-top) + 1/tail + 1/(df.shape[0]-num-tail))
        errors.append(error)
     
    # Build the plot
    fig, sub = plt.subplots(figsize=(15, 7))
    
    sub.bar(features, oddsratios,yerr=errors, align='center', alpha=0.5, ecolor='black', capsize=5)
   
    plt.ylabel("log2 enrichment of proband denovo data")
    
    plt.title("enrichment test, enrichment level = %s percent"%portion)
    
    plt.legend(features)
    plt.show()
              

def confusionmatrix(correct, predict, cutoff):
    predict_binary = np.copy(predict)
    predict_binary[predict_binary > cutoff ] = 1
    predict_binary[predict_binary <=  cutoff ] = 0
    cm = confusion_matrix(correct, predict_binary)
    print(cm)
    pl.matshow(cm)
    pl.title('Confusion matrix of the classifier')
    pl.colorbar()
    pl.show()
    
    
    
    
    
def savesemicol(df, col):
    col_max = []
    for i in df[col].str.split(';') :
        col_max.append(max(i))
        
    df[col] = np.array(col_max)
    return(df)

def save_allsemicol(df, columnname):
    for n in columnname:
        df = savesemicol(df, n)
        
    return(df)