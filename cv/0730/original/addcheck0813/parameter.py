import sys,subprocess,os
kernal = ["linear","rbf"]
#kernal = ['linear']
C = [1, 10,100]
gamma = [0.01, 0.1, 1, 10]
count_ = 0
for a in kernal:
    for b in C:
        for c in gamma:
            count_ += 1
            #cmd = 'bash tuning.sh %s %s %.3f %s'%(a,b,c, count_)
            cmd = 'qsub tuning.sh -F "%s %s %.3f %s"'%(a,b,c, count_)
            print(cmd)
            os.system(cmd)
            

            
           
