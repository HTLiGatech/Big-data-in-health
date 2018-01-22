import models_partc
from sklearn.cross_validation import KFold, ShuffleSplit
from numpy import mean

import utils

# USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

# USE THIS RANDOM STATE FOR ALL OF YOUR CROSS
# VALIDATION TESTS OR THE TESTS WILL NEVER PASS
RANDOM_STATE = 545510477

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y,k=5):
	#TODO:First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the folds
	length = X.get_shape()[0]
	kf = KFold(n = length, n_folds = k, random_state = RANDOM_STATE)
	accuracies = []
	AUCs = []

	for ktrain, ktest in kf:
		Xtrain = X[ktrain]
		Xtest = X[ktest]
		Ytrain = Y[ktrain]
		Ytest = Y[ktest]

		Ypredict = models_partc.logistic_regression_pred(Xtrain, Ytrain, Xtest)
		accuracy, AUC, _, _, _ = models_partc.classification_metrics(Ypredict, Ytest) 

		accuracies.append(accuracy)
		AUCs.append(AUC)

	return sum(accuracies) / k, sum(AUCs) / k



#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
	#TODO: First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the iterations
	length = X.get_shape()[0]
	sf = ShuffleSplit(n = length, n_iter = iterNo, test_size = 0.2, random_state = RANDOM_STATE)
	accuracies = []
	AUCs = []

	for ktrain, ktest in sf:
		Xtrain = X[ktrain]
		Xtest = X[ktest]
		Ytrain = Y[ktrain]
		Ytest = Y[ktest]

		Ypredict = models_partc.logistic_regression_pred(Xtrain, Ytrain, Xtest)
		accuracy, AUC, _, _, _ = models_partc.classification_metrics(Ypredict, Ytest) 

		accuracies.append(accuracy)
		AUCs.append(AUC)

	return sum(accuracies) / iterNo, sum(AUCs) / iterNo


def main():
	X,Y = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	print "Classifier: Logistic Regression__________"
	acc_k,auc_k = get_acc_auc_kfold(X,Y)
	print "Average Accuracy in KFold CV: "+str(acc_k)
	print "Average AUC in KFold CV: "+str(auc_k)
	acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
	print "Average Accuracy in Randomised CV: "+str(acc_r)
	print "Average AUC in Randomised CV: "+str(auc_r)

if __name__ == "__main__":
	main()

