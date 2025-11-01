import tensorflow as tf
import numpy as np
import pandas as pd

# machine learning
import keras
import ml_edu.experiment

#DEFİNE ML FUNCTİONS 
def create_model(
    settings:ml_edu.experiment.ExperimentSettings,
    metrics:list[keras.metrics.Metric],
    )->keras.Model:
    """Create and compile a simple linear regression model.""" 
    # Describe the topography of the model.
    # The topography of a simple linear regression model
    # is a single node in a single layer. 
    inputs={name:keras.Input(shape=(1,),name=name) for name in settings.input_features}
    concatenated_inputs=keras.layers.Concatenate()(list(inputs.values())) 
    outputs=keras.layers.Dense(units=1)(concatenated_inputs)
    model=keras.Model(inputs=inputs,outputs=outputs) 


    #Compile the model topography into that Keras can efficiently execute.
    #Configure training to minimize the model's mean squared error.
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
                  loss="mean_squared_error",
                  metrics=metrics)
    
    return model


def train_model(
        experiment_name: str,
        model: keras.Model,
        dataset: pd.DataFrame,
        label_name: str,
        settings: ml_edu.experiment.ExperimentSettings,
)->ml_edu.experiment.Experiment:
    """Train the model by feeding data."""

    #Feed the model feature and label.
    #The model will train for the specified number of epochs.
    features={name: dataset[name].values for name in settings.input_features}
    label=dataset[label_name].values
    history=model.fit(x=features,
                      y=label,
                      batch_size=settings.batch_size,
                      epochs=settings.number_epochs)
    return ml_edu.experiment.Experiment(name=experiment_name,
                                        model=model,
                                        settings=settings,
                                        epochs=history.epoch,
                                        metrics_history=pd.DataFrame(history.history),)
print("SUCCESS: defining linear regression functions complete.")

#DEFİNE FUNCTİONS TO MAKE PREDİCTİONS

def format_currency(x):
    return "${:.2f}".format(x)

def build_batch(df,batch_size):
    batch=df.sample(n=batch_size).copy()
    batch.set_index(np.arange(batch_size),inplace=True)
    return batch

def predict_fare(model,df,features,label,batch_size=50):
    batch=build_batch(df,batch_size)
    predicted_values=model.predict_on_batch(x={name: batch[name].values for name in features})

    data={"PREDICTED_FARE":[],"OBSERVED_FARE":[],"L1_LOSS":[],
          features[0]:[],features[1]:[]}
    
    for i in range(batch_size):
        predicted=predicted_values[i][0]
        observed=batch.at[i,label]
        data["PREDICTED_FARE"].append(format_currency(predicted))
        data["OBSERVED_FARE"].append(format_currency(observed))
        data["L1_LOSS"].append(format_currency(abs(observed-predicted)))
        data[features[0]].append(batch.at[i,features[0]])
        data[features[1]].append("{:.2f}".format(batch.at[i,features[1]]))

    output_df=pd.DataFrame(data)
    return output_df    

def show_predictions(output):
    header="_"*80
    banner=header+"\n"+"|"+"PREDICTIONS".center(78)+"|"+"\n"+header
    print(banner)
    print(output)
    return