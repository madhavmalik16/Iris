#ensemble code
import pandas
import csv

file1=pandas.read_csv('SVM.csv')
file2=pandas.read_csv('AdaBoostClassifier.csv')
file3=pandas.read_csv('NearestCentroid.csv')
file4=pandas.read_csv('RandomForestClassifier.csv')
file5=pandas.read_csv('MLPClassifier.csv')
file6=pandas.read_csv('KNeighborsClassifier.csv')
file7=pandas.read_csv('GaussianProcessClassifier.csv')
file8=pandas.read_csv('GaussianNB.csv')
file9=pandas.read_csv('QuadraticDiscriminantAnalysis.csv')
file10=pandas.read_csv('LogisticRegression.csv')

length=len(file1)   #can take any file i have taken file1
answer=file1    #default file can take any i have taken file1
for i in range(0,length):
	count_setosa=0
	count_versicolor=0
	count_virginica=0

	if(file1.iloc[i][0]==0):
		count_setosa=count_setosa+1
	if(file1.iloc[i][0]==1):
		count_versicolor=count_versicolor+1
	if(file1.iloc[i][0]==2):
		count_virginica=count_virginica+1

	if(file2.iloc[i][0]==0):
		count_setosa=count_setosa+1
	if(file2.iloc[i][0]==1):
		count_versicolor=count_versicolor+1
	if(file2.iloc[i][0]==2):
		count_virginica=count_virginica+1

	if(file3.iloc[i][0]==0):
		count_setosa=count_setosa+1
	if(file3.iloc[i][0]==1):
		count_versicolor=count_versicolor+1
	if(file3.iloc[i][0]==2):
		count_virginica=count_virginica+1

	if(file4.iloc[i][0]==0):
		count_setosa=count_setosa+1
	if(file4.iloc[i][0]==1):
		count_versicolor=count_versicolor+1
	if(file4.iloc[i][0]==2):
		count_virginica=count_virginica+1

	if(file5.iloc[i][0]==0):
		count_setosa=count_setosa+1
	if(file5.iloc[i][0]==1):
		count_versicolor=count_versicolor+1
	if(file5.iloc[i][0]==2):
		count_virginica=count_virginica+1

	if(file6.iloc[i][0]==0):
		count_setosa=count_setosa+1
	if(file6.iloc[i][0]==1):
		count_versicolor=count_versicolor+1
	if(file6.iloc[i][0]==2):
		count_virginica=count_virginica+1

	if(file7.iloc[i][0]==0):
		count_setosa=count_setosa+1
	if(file7.iloc[i][0]==1):
		count_versicolor=count_versicolor+1
	if(file7.iloc[i][0]==2):
		count_virginica=count_virginica+1

	if(file8.iloc[i][0]==0):
		count_setosa=count_setosa+1
	if(file8.iloc[i][0]==1):
		count_versicolor=count_versicolor+1
	if(file8.iloc[i][0]==2):
		count_virginica=count_virginica+1

	if(file9.iloc[i][0]==0):
		count_setosa=count_setosa+1
	if(file9.iloc[i][0]==1):
		count_versicolor=count_versicolor+1
	if(file9.iloc[i][0]==2):
		count_virginica=count_virginica+1

	if(file10.iloc[i][0]==0):
		count_setosa=count_setosa+1
	if(file10.iloc[i][0]==1):
		count_versicolor=count_versicolor+1
	if(file10.iloc[i][0]==2):
		count_virginica=count_virginica+1


	#print(count_setosa, count_versicolor, count_virginica)


	if(count_setosa>count_versicolor and count_setosa>count_virginica):
		answer.iloc[i][0]=0    #Iris-setosa
	if(count_versicolor>count_setosa and count_versicolor>count_virginica):
		answer.iloc[i][0]=1  #Iris-versicolor
	if(count_virginica>=count_versicolor and count_virginica>=count_setosa):
		answer.iloc[i][0]=2  #Iris-virginica

answer.to_csv("EnsembleResult.csv",index=False)


	



	
		