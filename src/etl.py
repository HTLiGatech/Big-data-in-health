import utils
import numpy as np
import pandas as pd
import datetime

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
	
	'''
	TODO: This function needs to be completed.
	Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
	
	Return events, mortality and feature_map
	'''

	#Columns in events.csv - patient_id,event_id,event_description,timestamp,value
	events = pd.read_csv(filepath + 'events.csv')
	
	#Columns in mortality_event.csv - patient_id,timestamp,label
	mortality = pd.read_csv(filepath + 'mortality_events.csv')

	#Columns in event_feature_map.csv - idx,event_id
	feature_map = pd.read_csv(filepath + 'event_feature_map.csv')

	return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
	
	'''
	TODO: This function needs to be completed.

	Refer to instructions in Q3 a

	Suggested steps:
	1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
	2. Split events into two groups based on whether the patient is alive or deceased
	3. Calculate index date for each patient
	
	IMPORTANT:
	Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
	Use the global variable deliverables_path while specifying the filepath. 
	Each row is of the form patient_id, indx_date.
	The csv file should have a header 
	For example if you are using Pandas, you could write: 
		indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

	Return indx_date
	'''

	indx_date = pd.DataFrame()
	alive = events[~events.patient_id.isin(mortality.patient_id)]
	alive_reset = alive.groupby(['patient_id']).agg({'timestamp' : np.max}).reset_index()

	indx_date['patient_id'] = mortality['patient_id']

	indx_date['timestamp'] = mortality.timestamp.apply(lambda x: (datetime.datetime.strptime(x, "%Y-%m-%d") - datetime.timedelta(days = 30)).strftime("%Y-%m-%d"))

	indx_date = pd.concat([alive_reset, indx_date]).reset_index().drop('index', 1)
	indx_date['timestamp'] = indx_date.timestamp.apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))

	indx_date = indx_date.rename(columns = {'timestamp' : 'indx_date'})

	indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)
	return indx_date




def filter_events(events, indx_date, deliverables_path):
	
	'''
	TODO: This function needs to be completed.

	Refer to instructions in Q3 b

	Suggested steps:
	1. Join indx_date with events on patient_id
	2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
	
	
	IMPORTANT:
	Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
	Use the global variable deliverables_path while specifying the filepath. 
	Each row is of the form patient_id, event_id, value.
	The csv file should have a header 
	For example if you are using Pandas, you could write: 
		filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

	Return filtered_events
	'''


	filtered_events = pd.merge(events, indx_date, on='patient_id')
	
	filtered_events['early'] = filtered_events.indx_date.apply(lambda x: x - datetime.timedelta(days = 2000))
	filtered_events['mid'] = filtered_events.timestamp.apply(lambda x: utils.date_convert(x))

	filtered_events = filtered_events[(filtered_events['mid'] <= filtered_events['indx_date']) & (filtered_events['mid'] >= filtered_events['early'])]
	filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

	return filtered_events


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
	
	'''
	TODO: This function needs to be completed.

	Refer to instructions in Q3 c

	Suggested steps:
	1. Replace event_id's with index available in event_feature_map.csv
	2. Remove events with n/a values
	3. Aggregate events using sum and count to calculate feature value
	4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
	
	
	IMPORTANT:
	Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
	Use the global variable deliverables_path while specifying the filepath. 
	Each row is of the form patient_id, event_id, value.
	The csv file should have a header .
	For example if you are using Pandas, you could write: 
		aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

	Return filtered_events
	'''
	aggregated = pd.merge(filtered_events_df, feature_map_df, on = 'event_id').dropna(subset = ['value'])
	aggregated = aggregated.groupby(['idx', 'patient_id', 'event_id'])
	aggregated = aggregated.agg({'value': [np.sum, np.mean, len, np.min, np.max]}).reset_index()

	aggregated['dmevent'] = aggregated['event_id'].apply(lambda x: x[0:3])

	aggregated['real'] = aggregated['value']['sum']

	aggregated.ix[aggregated['dmevent'] == 'LAB', 'real'] = list(aggregated.loc[aggregated['dmevent'] == 'LAB']['value']['len'])

	aggregated_events = pd.DataFrame()

	aggregated_events['value'] = aggregated['real']
	aggregated_events['feature_id'] = aggregated['idx']
	aggregated_events['patient_id'] = aggregated['patient_id']

	aggregated_events = aggregated_events.rename(columns={'idx': 'feature_id', 'actual_value' : 'value'})
	aggregated_events['max'] = aggregated_events.groupby('feature_id')['value'].transform(max)

	aggregated_events['normalize'] = aggregated_events['value'].divide(aggregated_events['max'],0)

	aggregated_events = aggregated_events.rename(columns = {'normalize': 'feature_value'})

	aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)
	return aggregated_events

def create_features(events, mortality, feature_map):
	
	deliverables_path = '../deliverables/'

	#Calculate index date
	indx_date = calculate_index_date(events, mortality, deliverables_path)

	#Filter events in the observation window
	filtered_events = filter_events(events, indx_date,  deliverables_path)
	
	#Aggregate the event values for each patient 
	aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

	'''
	TODO: Complete the code below by creating two dictionaries - 
	1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
	2. mortality : Key - patient_id and value is mortality label
	'''
	patient_features = {}
	mortalitydict = {}

	for i in range(len(mortality)):
		mortalitydict[float(mortality.ix[i, 'patient_id'])] = float(mortality.ix[i, 'label'])

	for i in range(len(aggregated_events)):
		if aggregated_events.ix[i, 'patient_id'] not in mortalitydict:
			mortalitydict[float(aggregated_events.ix[i, 'patient_id'])] = 0

		if aggregated_events.ix[i, 'patient_id'] not in patient_features:
			patient_features[float(aggregated_events.ix[i, 'patient_id'])] = [(float(aggregated_events.ix[i, 'feature_id']), float(aggregated_events.ix[i, 'feature_value']))]
		else:
			patient_features[float(aggregated_events.ix[i, 'patient_id'])].append((float(aggregated_events.ix[i, 'feature_id']), float(aggregated_events.ix[i, 'feature_value'])))

	return patient_features, mortalitydict

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
	
	'''
	TODO: This function needs to be completed

	Refer to instructions in Q3 d

	Create two files:
	1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
	2. op_deliverable - which saves the features in following format:
	   patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
	   patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
	
	Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.	 
	'''
	deliverable1 = open(op_file, 'wb')
	deliverable2 = open(op_deliverable, 'wb')

	for item in patient_features:
		newstr = ''
		if mortality[item] == 1:
			newstr += '1 '
		else:
			newstr += '0 '
		for ele in patient_features[item]:
			newstr += str(int(ele[0])) + ':' + ("%.6f" % float(ele[1])) + ' '
		deliverable1.write(newstr)
		deliverable1.write('\n')

	for item in patient_features:
		newstr = str(int(item)) + ' '
		if mortality[item] == 1:
			newstr += '1 '
		else:
			newstr += '0 '

		index = sorted(patient_features[item], key=lambda x: x[0])
		for ele in index:
			newstr += str(int(ele[0])) + ':' + ("%.6f" % float(ele[1])) + ' '
		deliverable2.write(newstr)
		deliverable2.write('\n')


def main():
	train_path = '../data/train/'
	events, mortality, feature_map = read_csv(train_path)
	patient_features, mortality = create_features(events, mortality, feature_map)
	save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
	main()