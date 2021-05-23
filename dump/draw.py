import matplotlib.pyplot as plt 
import re 
with open('dump/log.txt') as f:
    lst = f.readlines()
    epoches = []
    losses = []
    accs = []
    for i,l in enumerate(lst):
        if i%2==1:
            s = str(l)
            s = s.replace("epoch","")
            s = s.replace("loss","")
            s = s.replace("acc","")
            s = s.replace("tensor","")
            s = s.replace('(','')
            s = s.replace(')','')
            s = s.replace('\n','')
            nums = s.split('  ')
            
            epoches.append(int(nums[0]))
            losses.append(float(nums[1]))
            accs.append(float(nums[2]))

    print(losses)    
    plt.plot(losses)
    plt.plot(accs)
    plt.legend(['loss','acc'])
    plt.savefig('dump/curve.png')