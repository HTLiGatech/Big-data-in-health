import utils
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import KFold, ShuffleSplit
import pandas as pd
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import *
#Note: You can reuse code that you wrote in etl.py and models.py and cross.py over here. It might help.
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

'''
You may generate your own features over here.
Note that for the test data, all events are already filtered such that they fall in the observation window of their respective patients. Thus, if you were to generate features similar to those you constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.
IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
patient_id followed by a space and the corresponding feature in sparse format.
Eg of a line:
60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

Save the file as "test_features.txt" and save it inside the folder deliverables

input:
output: X_train,Y_train,X_test
'''
RANDOM_STATE = 545510477
def my_features():
	#TODO: complete this
	

	tests = pd.read_csv('../data/test/events.csv')
	feature = pd.read_csv('../data/test/event_feature_map.csv')
	new_features(tests, feature)

	X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	X_test, Y_test = utils.get_data_from_svmlight("../data/test/features_svmlight.test")


	return X_train, Y_train, X_test

def new_features(tests, feature):
	aggregated = pd.merge(tests, feature, on = 'event_id').dropna(subset = ['value'])
	aggregated = aggregated.groupby(['idx', 'patient_id', 'event_id'])
	aggregated = aggregated.agg({'value': [np.sum, np.mean, len, np.min, np.max]}).reset_index()

	aggregated_events = pd.DataFrame()
	aggregated_events['feature_id'] = aggregated['idx']
	aggregated_events['patient_id'] = aggregated['patient_id']
	aggregated_events['value'] = aggregated['value']['sum']

	aggregated_events['max'] = aggregated_events.groupby('feature_id')['value'].transform(max)
	aggregated_events['normalize'] = aggregated_events['value'].divide(aggregated_events['max'],0)
	aggregated_events = aggregated_events.rename(columns = {'normalize': 'feature_value'})

	patient_features = {}
	for i in range(len(aggregated_events)):
		if aggregated_events.ix[i, 'patient_id'] not in patient_features:
			patient_features[float(aggregated_events.ix[i, 'patient_id'])] = [(float(aggregated_events.ix[i, 'feature_id']), float(aggregated_events.ix[i, 'feature_value']))]
		else:
			patient_features[float(aggregated_events.ix[i, 'patient_id'])].append((float(aggregated_events.ix[i, 'feature_id']), float(aggregated_events.ix[i, 'feature_value'])))

	deliverable1 = open('../data/test/features_svmlight.test', 'w')
	deliverable2 = open('../deliverables/test_features.txt', 'w')
	
	for item in patient_features:
		newstr = '0 '
		for ele in patient_features[item]:
			newstr += str(int(ele[0])) + ':' + str(float(ele[1])) + ' '
		deliverable1.write(newstr)
		deliverable1.write('\n')

	for item in patient_features:
		newstr = str(int(item)) + ' '
		index = sorted(patient_features[item], key=lambda x: x[0])
		for ele in index:
			newstr += str(int(ele[0])) + ':' + str(float(ele[1])) + ' '
		deliverable2.write(newstr)
		deliverable2.write('\n')

'''
You can use any model you wish.

input: X_train, Y_train, X_test
output: Y_pred
'''
# def my_classifier_predictions(X_train,Y_train,X_test):
# 	#TODO: complete this
# 	ada = AdaBoostClassifier(n_estimators=38, random_state=RANDOM_STATE)
# 	ada.fit(X_train, Y_train)
# 	return ada.predict(X_test)

def my_classifier_predictions(X_train,Y_train,X_test):
	c = 0.01

	svm = LinearSVC(C = c, random_state = RANDOM_STATE)
	svm.fit(X_train, Y_train)
	return svm.predict(X_test)

#input: Y_pred,Y_true
#output: accuracy, auc, precision, recall, f1-score


def main():
	X_train, Y_train, X_test = my_features()
	Y_pred = my_classifier_predictions(X_train,Y_train,X_test)

	utils.generate_submission("../deliverables/test_features.txt",Y_pred)
	#The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.

if __name__ == "__main__":
	main()

	