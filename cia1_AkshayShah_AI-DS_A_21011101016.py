import random
import numpy as np
list = ['A','G','C','T']
list2 = []
list3 = []
a = 0
g = 0
c = 0
t = 0
for i in list2:
    if i=='A':
        a+=1
    elif i=='G':
        g+=1
    elif i=='T':
        t+=1
    elif i=='C':
        c+=1
        

while(True):
    h = random.choice(list)
    if len(list2)==16:
        break
    if h=='A':
        if a>=4:
            continue
        else:
            list2.append(h)
    elif h=='G':
        if g>=4:
            continue
        else:
            list2.append(h)
    elif h=='C':
        if c>=4:
            continue
        else:
            list2.append(h)
    elif h=='T':
        if t>=4:
            continue
        else:
            list2.append(h)

s1 = ''.join(list2)
print(s1)


a1= 0
g1 = 0
c1 = 0
t1 = 0
for i in list3:
    if i=='A':
        a1+=1
    elif i=='G':
        g1+=1
    elif i=='T':
        t1+=1
    elif i=='C':
        c1+=1
        

while(True):
    j = random.choice(list)
    if len(list3)==16:
        break
    if j=='A':
        if a1>=4:
            continue
        else:
            list3.append(j)
    elif j=='G':
        if g1>=4:
            continue
        else:
            list3.append(j)
    elif j=='C':
        if c1>=4:
            continue
        else:
            list3.append(j)
    elif j=='T':
        if t1>=4:
            continue
        else:
            list3.append(j)

s2 = ''.join(list3)
print(s2)

match = 5
mismatch = 4
len1,len2 = 16,16
zeroes = np.zeros((16,16),dtype = 'int')
i = 1
j = 1
while(True):
    if i==len1 and j==len2:
        print(zeroes)
        break
    if s1[i-1]!=s2[j-1]:
        x = max(zeroes[i][j-1],zeroes[i-1][j],zeroes[i-1][j-1])
        zeroes[i][j] = x-mismatch
    else:
        x = zeroes[i-1][j-1]+ match
        zeroes[i][j] = x
    if j==len2:
        i+=1
        j = 1
    else:
        j+=1






    
            
