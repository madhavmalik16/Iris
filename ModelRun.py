import csv
import pandas
import sklearn   #we use the model_selection module in the sklearn library
from sklearn import model_selection
from sklearn import svm   #for SVM
from sklearn.ensemble import AdaBoostClassifier #for Adaboost
from sklearn.tree import DecisionTreeClassifier #for Adaboost
from sklearn.neighbors.nearest_centroid import NearestCentroid #for K Nearest Neighbour Centroid
from sklearn.ensemble import RandomForestClassifier #for Random Forest Classifier
from sklearn.neural_network import MLPClassifier #for MLP Classifier
from sklearn.neighbors import KNeighborsClassifier #for KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier #for Gaussian Process Classifier
from sklearn.naive_bayes import GaussianNB #for Gaussian Naive Bayes
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis #for Quadratic Discriminant Analysis
from sklearn.linear_model import LogisticRegression #for Logistic Regression
import matplotlib
import scipy
import sys

def model_runner(model):
	model_name=model
	model_name.fit(X_train,Y_train)
	#resultTrain=svm1.predict(X_train)
	resultTest=model_name.predict(X_validation)

	count=0
	for i in range(0,45):
		if(Y_validation[i]==resultTest[i]):
			count=count+1

	total=len(resultTest)
	accuracy=(count/total)*100
	print(accuracy,"%")




with open('iris.csv','r') as file:
	iris=pandas.read_csv(file)
	
	#printing the number of rows and columns in the dataset
	#print(iris.shape)


	
	array=iris.values
	X=array[:,0:4]   #This basically means that first dimension is completely incorporated and first four values of the second dimension is incorporated
	Y=array[:,4] #This picks up the fourth coloumn of the dataset


	#Splitting the dataset for 70-30 training,testing
	validation_size=0.30
	#conventional value of seed is 7 but can take any
	seed=5

	X_train, X_validation, Y_train, Y_validation=model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


	#Calling models
	#Model Number 1 SVM
	svm1=svm.SVC()
	model_runner(svm1)

	#Model Number 2 Adaboost
	ada1=AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),n_estimators=300)
	model_runner(ada1)

	#Model Number 3 K Nearest Neighbour
	knn1=NearestCentroid()
	model_runner(knn1)

	#Model Number 4 Random Forest Classifier
	rfc1=RandomForestClassifier(max_depth=2,random_state=0)
	model_runner(rfc1)

	#Model Number 5 MLP Classifier
	mlp1=MLPClassifier(hidden_layer_sizes=(100,), activation="relu",solver='adam', alpha=0.0001,batch_size='auto', learning_rate="constant",learning_rate_init=0.001,power_t=0.5, max_iter=200, shuffle=True,random_state=0, tol=1e-4,verbose=False, warm_start=False, momentum=0.9,nesterovs_momentum=True, early_stopping=False,validation_fraction=0.1, beta_1=0.9, beta_2=0.999,epsilon=1e-8)
	model_runner(mlp1)  #This model runs with a convergence warning

	#Model Number 6 K Neighbour Classifier
	knc1=KNeighborsClassifier(n_neighbors=3)
	model_runner(knc1)

	#Model Number 7 Gaussian Process Classifier
	gpc1=GaussianProcessClassifier(kernel=None, n_restarts_optimizer=0, max_iter_predict=100, warm_start=False, copy_X_train=True, random_state=None, n_jobs=1)
	model_runner(gpc1)  #Remove optimizer=’fmin_l_bfgs_b’ and multi_class=’one_vs_rest’

	#Model Number 8 Gaussian Naive Bayes
	gnb1=GaussianNB()
	model_runner(gnb1)

	#Model Number 9 Quadratic Discriminant Analysis
	qda1=QuadraticDiscriminantAnalysis()
	model_runner(qda1)

	#Model Number 10 Logistic Regression
	lor1=LogisticRegression()
	model_runner(lor1)