#VİEW DATASET STATİSTİCS
import plotly_express as px
from read_dataset import get_prepared_dataframe
training_df=get_prepared_dataframe()

print('Total number of rows:{0}\n\n'.format(len(training_df.index)))
training_df.describe(include="all")
print(training_df.describe())


#What is the maximum fare?
max_fare=training_df['FARE'].max()
print('What is the maximum fare?     Answer: ${fare:.2f}'.format(fare=max_fare))


#What is the mean distance across all trips?
mean_distance=training_df['TRIP_MILES'].mean()
print('What is the mean distance across all trips?    Answer:{mean:.4f} miles'.format(mean=mean_distance))


#How many cab companies in the dataset?
num_unique_companies=training_df['COMPANY'].nunique()
print('How many cab companies in the dataset?    Answer:{number}'.format(number=num_unique_companies))


#What is the most frequent payment type?
most_frequent_payment_type=training_df['PAYMENT_TYPE'].value_counts().idxmax()
print('What is the most frequent payment type?    Answer:{type}'.format(type=most_frequent_payment_type))


#Are any features missing data?
missing_values=training_df.isnull().sum().sum()
print('Are any features missing data?   Answer:',"No" if missing_values==0 else "Yes")

#VİEW CORROLATİON MATRİX
training_df.corr(numeric_only=True)
print(training_df.corr(numeric_only=True))
answer = '''
The feature with the strongest correlation to the FARE is TRIP_MILES.
As you might expect, TRIP_MILES looks like a good feature to start with to train
the model. Also, notice that the feature TRIP_SECONDS has a strong correlation
with fare too.
'''
print(answer)


# Which feature correlates least strongly to the label FARE?
# -----------------------------------------------------------
answer = '''The feature with the weakest correlation to the FARE is TIP_RATE.'''
print(answer)

#VİEW PAİRPLOT
fig=px.scatter_matrix(training_df,dimensions=["FARE","TRIP_MILES","TRIP_SECONDS"])
fig.show()