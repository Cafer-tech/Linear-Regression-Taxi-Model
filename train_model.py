# machine learning
import keras
import ml_edu.experiment
import ml_edu.results

import matplotlib.pyplot as plt

from define_functions import create_model
from define_functions import train_model
from read_dataset import training_df

settings_1=ml_edu.experiment.ExperimentSettings(
    learning_rate=0.001,
    number_epochs=20,
    batch_size=50,
    input_features=["TRIP_MILES","TRIP_SECONDS"]
)

metrics=[keras.metrics.RootMeanSquaredError(name='rmse')]

model_1=create_model(settings_1,metrics)

experiment_1=train_model('Two_Features',model_1,training_df,'FARE',settings_1)
#SAVE THE MODEL
experiment_1.model.save("linear_regression_taxi.h5")

ml_edu.results.plot_experiment_metrics(experiment_1,['rmse'])
ml_edu.results.plot_model_predictions(experiment_1,training_df,'FARE')
plt.show()