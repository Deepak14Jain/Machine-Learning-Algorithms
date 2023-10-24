import os
import pandas as pd
import numpy as np

os.chdir("Documents")
a = []
f = pd.read_csv("enjoysport.csv")
for row in f.iterrows():
    a.append(row[1])
num_attributes = len(a[0])
num_instances = len(a)
temp = [0]
hypothesis = (num_attributes-1)*temp
for i in range(num_instances):
    if a[i][num_attributes-1] == "yes":
        for j in range(num_attributes-1):
            if (hypothesis[j] == 0) or (hypothesis[j] == a[i][j]):
                hypothesis[j] = a[i][j]
            else:
                hypothesis[j] = '?'
print("The most specific hypothesis for the given dataset is: ",hypothesis)
