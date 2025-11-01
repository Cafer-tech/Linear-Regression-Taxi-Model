import keras
from define_functions import predict_fare
from define_functions import show_predictions
from read_dataset import training_df

#LOAD THE MODEL
loaded_model=keras.models.load_model('linear_regression_taxi.h5')

predicted_features=["TRIP_MILES","TRIP_SECONDS"]


output=predict_fare(loaded_model,training_df,predicted_features,"FARE")
show_predictions(output)