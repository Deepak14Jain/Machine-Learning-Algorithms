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
temp_s = [0]
temp_g = ['?']
s_hypothesis = (num_attributes-1)*temp_s
hypothesis = []
g_hypothesis = []
for i in range(num_instances):
    if a[i][num_attributes-1] == "yes":
        for j in range(num_attributes-1):
            if (s_hypothesis[j] == 0) or (s_hypothesis[j] == a[i][j]):
                s_hypothesis[j] = a[i][j]
            else:
                s_hypothesis[j] = '?'
    else:
        for j in range(num_attributes-1):
            for k in range(num_attributes-1):
                if (s_hypothesis[j] != a[i][k]) and (s_hypothesis[j] != '?') and (j==k):
                    hypothesis.append(s_hypothesis[j])
                else:
                    hypothesis.append('?')
            if hypothesis not in g_hypothesis:
                g_hypothesis.append(hypothesis)
            hypothesis = []
for k in range(len(g_hypothesis)):
    for z in range(num_attributes-1):
        if (g_hypothesis[k][z] != '?') and (g_hypothesis[k][z] != s_hypothesis[z]):
            g_hypothesis.pop(k)
temp = (num_attributes-1)*temp_g
if temp in g_hypothesis:
    g_hypothesis.remove(temp)
print("The most specific hypothesis for the given dataset is: ",s_hypothesis)
print("The most general hypothesis for the given dataset is: ",g_hypothesis)
