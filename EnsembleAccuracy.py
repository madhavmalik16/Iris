#Final Accuracy
#Came out to be around 53.333333336
#Conclusion:- Individual models have better accuracy, Ensembling not that useful for my train-test split of iris dataset

import pandas
import csv

file1=pandas.read_csv('finalTester.csv')
file2=pandas.read_csv('EnsembleResult.csv')

count=0
total=45
for i in range(0,45):
	if(file1.iloc[i][0]==file2.iloc[i][0]):
		count=count+1
accuracy=(count/45)*100
print(accuracy)