# Name this file assignment2.py when you submit
import numpy as np
import keras
from keras import Sequential, Model, layers, optimizers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from statistics import mean

from matplotlib import pyplot as plt
# A function that implements a keras model with the sequential API following the provided description
def sequential_model(): #layers are 8, 16, 12, 8, 4
  model = Sequential()
  model.add(layers.Dense(8, input_dim=8, activation='relu'))
  model.add(layers.Dense(16, activation='relu'))
  model.add(layers.Dense(12, activation='relu'))
  model.add(layers.Dense(8, activation='relu'))
  model.add(layers.Dense(4, activation='softmax'))
  model.optimizer = 'SGD'
  model.loss = 'categorical_crossentropy'
  # A keras model
  return model


# A function that implements a keras model with the functional API following the provided description

def functional_model():# layers are 4, 8, 2, 2
  inputs = layers.Input(shape=(4,))
  
  hidden1 = layers.Dense(8, activation='relu')(inputs)
  hidden2 = layers.Dense(2, activation='relu')(hidden1)
  
  output1 = layers.Dense(2, activation='softmax')(hidden2)
  output2 = layers.Dense(2, activation='softmax')(hidden2)
  
  model = Model(inputs=inputs, outputs=[output1, output2])
  model.compile(optimizer=optimizers.SGD(),
              loss={'task1': 'binary_crossentropy', 'task2': 'binary_crossentropy'},
              loss_weights={'task1': 1.0, 'task2': 1.0},  # Sum of losses
              metrics=['accuracy'])
  # A keras model
  return model


  # hidden1 = layers.Dense(128, activation='relu')(inputs)
  # hidden2 = layers.Dense(32, activation='relu')(hidden1)
  # hidden3 = layers.Dense(8, activation='relu')(hidden2)
  # hidden4 = layers.Dense(2, activation='relu')(hidden3)
  # these were the original hidden layers.
def thyroid_model_FFN(hyperparam_dict):
  layers_mask = hyperparam_dict['layers']
  neurons_hyperparams = hyperparam_dict['neurons']
  
  if layers_mask == None:
    layers_mask = [1, 0, 1, 0, 1, 0, 1] #if not adjusting parameters of the model, we keep layer 0, 2, 4, 6 on.
    
  if neurons_hyperparams == None:
    neurons_hyperparams = [128, 64, 32, 16, 8, 4, 2]
  print(hyperparam_dict)

  inputs = layers.Input(shape=(55,))
  
  previous_layer = inputs
  for i in range(len(layers_mask)):
    if layers_mask[i] == 0:
      continue
    next_layer = layers.Dense(neurons_hyperparams[i], activation='relu')(previous_layer)
    previous_layer = next_layer
    
  output = layers.Dense(1, activation='sigmoid')(previous_layer)
  
  model = Model(inputs=inputs, outputs=output)
  model.compile(optimizer=optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
  
  return model
  
def train_validate(model, X_train, y_train, X_val, y_val, epochs=10):
  age_mean = np.mean(X_train[0], axis=0) # make sure we standardize using X_train to prevent data leakage
  age_std = np.std(X_train[0], axis=0)
  
  X_train[0] = (X_train[0] - age_mean) / age_std #standardize and normalize age since it's the only numerical feature
  X_val[0] = (X_val[0] - age_mean) / age_std

  history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=32)
  validation_accuracy = history.history['val_accuracy'][-1]  # last validation accuracy
  
  return model, validation_accuracy, age_mean, age_std

def data_preprocessing(filepath):
  data = pd.read_csv(filepath)  
  # for col in data:
  #   print(f'Unique values for column "{col}": {data[col].unique()}')
  data['Recurred'] = data['Recurred'].apply(lambda x: 1 if x == 'Yes' else 0)
  data = pd.get_dummies(data, columns=data.columns[1:-1])
  data_matrix = data.to_numpy()
  raw_y = data_matrix[:, 1].astype('float32')
  raw_X = np.delete(data_matrix, 1, axis=1).astype('float32')
  return raw_X, raw_y

  
# A function that creates a keras model to predict whether a patient has recurrence of thryroid cancer
def thyroid_cancer_recurrence_model(filepath):
  # filepath is the path to an csv file containing the dataset
  raw_X, raw_y = data_preprocessing(filepath)    
  hyperparam_dict = {'layers': None, 'neurons': None}
  kf = KFold(n_splits=5, shuffle=True, random_state=42)
  accuracy_history = []
  for train_index, val_index in kf.split(raw_X):
    X_train, X_val = raw_X[train_index], raw_X[val_index]
    y_train, y_val = raw_y[train_index], raw_y[val_index]    
    model = thyroid_model_FFN(hyperparam_dict) # reset model
    
    model, validation_accuracy, _, _ = train_validate(model, X_train, y_train, X_val, y_val)
    accuracy_history.append(validation_accuracy)
    
  averaged_validation_accuracy = mean(accuracy_history)
  print(averaged_validation_accuracy)
  #final training
  # do several times to get best model
  age_mean_full = np.mean(raw_X[0], axis=0) # scale and normalize on entire dataset
  age_std_full = np.std(raw_X[0], axis=0)
  raw_X[0] = (raw_X[0] - age_mean_full) / age_std_full

  best_accuracy = 0
  best_model = None

  for _ in range(5):
    
    model = thyroid_model_FFN(hyperparam_dict)
    history = model.fit(raw_X, raw_y, epochs=10, batch_size=32)
    training_accuracy = history.history['accuracy'][-1]
    
    if validation_accuracy > best_accuracy:
      best_accuracy = training_accuracy
      best_model = model
  # model is a trained keras model for predicting whether a a patient has recurrence of thryroid cancer during a follow-up period
  # validation_performance is the performance of the model on a validation set

  return best_model, averaged_validation_accuracy

def thyroid_autoexperiments(filepath):
  #masks for adjusting the hidden layers, 1: turned on. 0: turned off
  raw_X, raw_y = data_preprocessing(filepath)
  test_ratio = 0.2
  X_train, X_test, y_train, y_test = train_test_split(raw_X, raw_y, test_size=test_ratio)
  kf = KFold(n_splits=5, shuffle=True, random_state=42)
  
  
  hidden_layers_adjustment = [
    [0,0,0,0,0,0,0], #remove all layers
    [1,0,0,0,0,0,0], 
    [1,0,1,0,0,0,0],
    [1,0,1,0,1,0,0],
    [1,0,1,0,1,0,1], #original model
    [1,1,1,0,1,0,1], #start to interleave layers
    [1,1,1,1,1,0,1],
    [1,1,1,1,1,1,1] 
                              ]
  best_accuracy = 0
  best_model = None
  best_mask = None
  mask_acc_history = []
  
  for mask in hidden_layers_adjustment:
    hyperparam_dict = {'layers': mask, 'neurons': None}
    accuracy_history = []
    
    for train_index, val_index in kf.split(raw_X):
      X_train, X_val = raw_X[train_index], raw_X[val_index]
      y_train, y_val = raw_y[train_index], raw_y[val_index]    
      model = thyroid_model_FFN(hyperparam_dict) # reset model
      
      model, validation_accuracy, _, _ = train_validate(model, X_train, y_train, X_val, y_val)
      accuracy_history.append(validation_accuracy)
      if best_accuracy < validation_accuracy:
        best_accuracy = validation_accuracy
        best_model = model
        best_mask = mask
      
    averaged_validation_accuracy = mean(accuracy_history)
    mask_acc_history.append(averaged_validation_accuracy)
    layers_used = np.array(mask) * np.array([128, 64, 32, 16, 8, 4, 2])

    print(f'Averaged validation accuracy for the model:{layers_used} is {averaged_validation_accuracy}')
  
  layers_used = np.array(best_mask) * np.array([128, 64, 32, 16, 8, 4, 2])
  print(f'Best model has validation accuracy of {best_accuracy} with mask {best_mask}')
  
  x = np.arange(len(hidden_layers_adjustment))
  y = np.array(mask_acc_history)
  plt.bar(x, y)
  plt.xlabel('Number of Layers')
  plt.ylabel('Averaged Validation Accuracy')
  plt.xticks(x, labels=[str(sum(mask)) for mask in hidden_layers_adjustment])
  plt.title('Validation Accuracy vs. Number of Layers')
  plt.savefig('layer_validation_accuracy.png')
        
  
  
  #hyperparameters for adjusting neurons in the hidden layers
  neuron_ratio_adjustment = [1/4, 1/2, 3/4, 1, 5/4, 3/2, 2]

  
  #hyperparameters for adjusting the epochs
  epochs_adjustment = [4, 8, 12, 16, 20, 24]
  
  
  
# model, validation_performance = thyroid_cancer_recurrence_model('Thyroid_Diff.csv')

thyroid_autoexperiments('Thyroid_Diff.csv')