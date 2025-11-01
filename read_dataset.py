import pandas as pd
from import1 import get_full_dataset
chicago_taxi_dataset=get_full_dataset()
#UPDATE THE DATAFRAME
training_df=chicago_taxi_dataset.loc[:,('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')]
print('Read dataset complicated successfully.')
print('Total number of rows:{0}\n\n'.format(len(training_df.index)))
training_df.head(200)
def get_prepared_dataframe():
    chicago_taxi_dataset=pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")
    training_df=chicago_taxi_dataset.loc[:,('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')]
    return training_df