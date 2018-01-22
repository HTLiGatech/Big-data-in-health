import time
import pandas as pd
import numpy as np
from datetime import datetime

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    '''
    TODO : This function needs to be completed.
    Read the events.csv and mortality_events.csv files. 
    Variables returned from this function are passed as input to the metric functions.
    '''
    events = pd.read_csv(filepath + 'events.csv')
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    return events, mortality

def event_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the event count metrics.
    Event count is defined as the number of events recorded for a given patient.
    '''
    dead_patients = pd.merge(events, mortality, on = 'patient_id')
    alive_patients = events[~events.patient_id.isin(dead_patients.patient_id)]



    count_dead = dead_patients.groupby(['patient_id']).count()
    count_alive = alive_patients.groupby(['patient_id']).count()

    avg_dead_event_count = count_dead['event_id'].mean()
    max_dead_event_count = count_dead['event_id'].max()
    min_dead_event_count = count_dead['event_id'].min()

    avg_alive_event_count = count_alive['event_id'].mean()
    max_alive_event_count = count_alive['event_id'].max()
    min_alive_event_count = count_alive['event_id'].min()

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the encounter count metrics.
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 

    '''

    dead_patients = pd.merge(events, mortality, on = 'patient_id')
    alive_patients = events[~events.patient_id.isin(dead_patients.patient_id)]

    unique_dead = dead_patients.groupby(['patient_id']).timestamp_x.nunique()
    unique_alive = alive_patients.groupby(['patient_id']).timestamp.nunique()

    print unique_dead

    avg_dead_encounter_count = unique_dead.mean()
    max_dead_encounter_count = unique_dead.max()
    min_dead_encounter_count = unique_dead.min()

    avg_alive_encounter_count = unique_alive.mean()
    max_alive_encounter_count = unique_alive.max()
    min_alive_encounter_count = unique_alive.min()


    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    TODO: Implement this function to return the record length metrics.
    Record length is the duration between the first event and the last event for a given patient. 
    '''
    dead_patients = pd.merge(events, mortality, on = 'patient_id')
    alive_patients = events[~events.patient_id.isin(dead_patients.patient_id)]

    date_dead = dead_patients.groupby(['patient_id']).timestamp_x
    date_alive = alive_patients.groupby(['patient_id']).timestamp

    max_dead_dates, min_dead_dates = list(date_dead.max()), list(date_dead.min())
    max_alive_dates, min_alive_dates = list(date_alive.max()), list(date_alive.min())
    date_diff_dead, date_diff_alive = [], []


    for i in range(len(max_dead_dates)):
    	diff = datetime.strptime(max_dead_dates[i], '%Y-%m-%d') - datetime.strptime(min_dead_dates[i], '%Y-%m-%d')
    	date_diff_dead.append(diff.days)
    for i in range(len(max_alive_dates)):
    	diff = datetime.strptime(max_alive_dates[i], '%Y-%m-%d') - datetime.strptime(min_alive_dates[i], '%Y-%m-%d')
    	date_diff_alive.append(diff.days)

    avg_dead_rec_len = float(sum(date_diff_dead))/len(date_diff_dead)
    max_dead_rec_len = max(date_diff_dead)
    min_dead_rec_len = min(date_diff_dead)
    avg_alive_rec_len = float(sum(date_diff_alive))/len(date_diff_alive)
    max_alive_rec_len = max(date_diff_alive)
    min_alive_rec_len = min(date_diff_alive)

    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    '''
    You may change the train_path variable to point to your train data directory.
    OTHER THAN THAT, DO NOT MODIFY THIS FUNCTION.
    '''
    # You may change the following line to point the train_path variable to your train data directory
    train_path = '../data/train/'

    # DO NOT CHANGE ANYTHING BELOW THIS ----------------------------
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute event count metrics: " + str(end_time - start_time) + "s")
    print event_count

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute encounter count metrics: " + str(end_time - start_time) + "s")
    print encounter_count

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute record length metrics: " + str(end_time - start_time) + "s")
    print record_length
    
if __name__ == "__main__":
    main()
