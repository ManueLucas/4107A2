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

  history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=32, verbose=0)
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
  X_trainval, X_test, y_trainval, y_test = train_test_split(raw_X, raw_y, test_size=test_ratio)
  kf = KFold(n_splits=5, shuffle=True, random_state=42)
  full_model_neurons = np.array([128, 64, 32, 16, 8, 4, 2])

  def layer_experiments():
    
    
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
    # validation measures
    best_accuracy = 0
    best_mask = None
    mask_acc_history = []
    best_training_indices_layer = None
    best_validation_indices_layer = None
    
    for mask in hidden_layers_adjustment:
      hyperparam_dict = {'layers': mask, 'neurons': None}
      accuracy_history = []
      
      for train_index, val_index in kf.split(X_trainval):
        X_train, X_val = X_trainval[train_index], X_trainval[val_index] #we split from trainval instead of raw to avoid data leakage
        y_train, y_val = y_trainval[train_index], y_trainval[val_index]    
        model = thyroid_model_FFN(hyperparam_dict) # reset model
        
        model, validation_accuracy, age_mean, age_std = train_validate(model, X_train, y_train, X_val, y_val)
        accuracy_history.append(validation_accuracy)
        if best_accuracy < validation_accuracy: #grab best model during validation phase
          best_accuracy = validation_accuracy
          model.save('best_layer_model.h5')
          best_mask = mask
          best_training_indices_layer = train_index
          best_validation_indices_layer = val_index
        
      averaged_validation_accuracy = mean(accuracy_history)
      mask_acc_history.append(averaged_validation_accuracy)
      layers_used = np.array(mask) * full_model_neurons

      print(f'Averaged validation accuracy for the model:{layers_used} is {averaged_validation_accuracy}')
    
    layers_used = np.array(best_mask) * full_model_neurons
    print(f'Best model has validation accuracy of {best_accuracy} with mask {best_mask}')
    
    x = np.arange(len(hidden_layers_adjustment))
    y = np.array(mask_acc_history)
    plt.clf()
    plt.bar(x, y)
    plt.xlabel('Number of Layers')
    plt.ylabel('Averaged Validation Accuracy')
    plt.xticks(x, labels=[str(sum(mask)) for mask in hidden_layers_adjustment])
    plt.title('Validation Accuracy vs. Number of Layers')
    plt.savefig('layer_validation_accuracy.png')
    return best_mask, (best_training_indices_layer, best_validation_indices_layer)
  
  
  #hyperparameters for adjusting neurons in the hidden layers
  def neuron_experiments():
    neuron_ratio_adjustment = [1/4, 1/2, 3/4, 1, 5/4, 3/2]
    best_accuracy = 0
    best_ratio = None
    neuron_acc_history = []
    hidden_neurons_history = []
    best_training_indices_neuron = None
    best_validation_indices_neuron = None
    
    for ratio in neuron_ratio_adjustment:
      neurons_hyperparams = np.ceil(full_model_neurons * ratio).astype(int).tolist()
      hidden_neurons_history.append(np.sum(neurons_hyperparams))
      
      hyperparam_dict = {'layers': None, 'neurons': neurons_hyperparams}
      accuracy_history = []
      
      for train_index, val_index in kf.split(X_trainval):
        X_train, X_val = X_trainval[train_index], X_trainval[val_index]
        y_train, y_val = y_trainval[train_index], y_trainval[val_index]    
        model = thyroid_model_FFN(hyperparam_dict)
        
        model, validation_accuracy, _, _ = train_validate(model, X_train, y_train, X_val, y_val)
        accuracy_history.append(validation_accuracy)
        if best_accuracy < validation_accuracy:
          best_accuracy = validation_accuracy
          model.save('best_neuron_model.h5')
          best_ratio = ratio
          best_training_indices_neuron = train_index
          best_validation_indices_neuron = val_index
          
      #k fold is done
      averaged_validation_accuracy = mean(accuracy_history)
      neuron_acc_history.append(averaged_validation_accuracy)
      print(f'Averaged validation accuracy for the ratio:{ratio} is {averaged_validation_accuracy}')
      #onto the next ratio...
    
    print(f'Best model has validation accuracy of {best_accuracy} with ratio {best_ratio}')

    x = np.array(hidden_neurons_history)
    y = np.array(neuron_acc_history)
    plt.clf()
    plt.bar(x, y, width=50)
    plt.xlabel('Hidden Layer Neurons')
    plt.ylabel('Averaged Validation Accuracy')
    plt.xticks(x, labels=[str(neurons) for neurons in hidden_neurons_history])
    plt.title('Validation Accuracy vs. Number of Total Neurons in Hidden Layers')
    plt.savefig('Neuron_validation_accuracy.png')
    return best_ratio, (best_training_indices_neuron, best_validation_indices_neuron)

  
  #hyperparameters for adjusting the epochs
  def epoch_experiments():
    epochs_adjustment = [4, 8, 12, 16, 20, 24]
    best_accuracy = 0
    epoch_acc_history = []
    best_training_indices_epoch = None
    best_validation_indices_epoch = None
    
    for epoch in epochs_adjustment:      
      hyperparam_dict = {'layers': None, 'neurons': None}
      accuracy_history = []
      
      for train_index, val_index in kf.split(X_trainval):
        X_train, X_val = X_trainval[train_index], X_trainval[val_index]
        y_train, y_val = y_trainval[train_index], y_trainval[val_index]    
        model = thyroid_model_FFN(hyperparam_dict)
        
        model, validation_accuracy, _, _ = train_validate(model, X_train, y_train, X_val, y_val, epochs=epoch)
        accuracy_history.append(validation_accuracy)
        if best_accuracy < validation_accuracy:
          best_accuracy = validation_accuracy
          model.save('best_epoch_model.h5')
          best_epoch = epoch
          best_training_indices_epoch = train_index
          best_validation_indices_epoch = val_index
          
      #k fold is done
      averaged_validation_accuracy = mean(accuracy_history)
      epoch_acc_history.append(averaged_validation_accuracy)
      print(f'Averaged validation accuracy for the epoch:{epoch} is {averaged_validation_accuracy}')
    
    print(f'Best model has validation accuracy of {best_accuracy} with epochs {best_epoch}')

    x = np.array(epochs_adjustment)
    y = np.array(epoch_acc_history)
    plt.clf()
    plt.bar(x, y, width=3)
    plt.xlabel('Epochs')
    plt.ylabel('Averaged Validation Accuracy')
    plt.xticks(x, labels=[str(epochs) for epochs in epochs_adjustment])
    plt.title('Validation Accuracy vs. # of Epochs Trained')
    plt.savefig('Epochs_validation_accuracy.png')
    
    return best_epoch, (best_training_indices_epoch, best_validation_indices_epoch)
  print('Starting layer experiments...')
  best_layer, layer_splits = layer_experiments()
  print('Starting neuron experiments...')
  best_neuron, neuron_splits = neuron_experiments()
  print('Starting epoch experiments...')
  best_epoch, epoch_splits = epoch_experiments()
  
  def train_validate_test_performance(layer_param_file, neuron_param_file, epoch_param_file, layer_splits, neuron_splits, epoch_splits, best_layer, best_neuron, best_epoch):
    best_layer_model = keras.models.load_model(layer_param_file)
    best_neuron_model = keras.models.load_model(neuron_param_file)
    best_epoch_model = keras.models.load_model(epoch_param_file)
    
    #test the best layer model
    X_train, X_val = X_trainval[layer_splits[0]], X_trainval[layer_splits[1]]
    y_train, y_val = y_trainval[layer_splits[0]], y_trainval[layer_splits[1]]
    
    print(f'With layer model {best_layer} ({sum(best_layer)} layers):')
    _, accuracy = best_layer_model.evaluate(X_train, y_train)
    print(f'Achieved training accuracy of {accuracy}')   
    _, accuracy = best_layer_model.evaluate(X_val, y_val)
    print(f'Achieved validation accuracy of {accuracy}')
    _, accuracy = best_layer_model.evaluate(X_test, y_test)
    print(f'Achieved test accuracy of {accuracy}')
    
    #test the best neuron model
    X_train, X_val = X_trainval[neuron_splits[0]], X_trainval[neuron_splits[1]]
    y_train, y_val = y_trainval[neuron_splits[0]], y_trainval[neuron_splits[1]]
    
    print(f'With hidden layer total parameter count of {best_neuron}:')
    _, accuracy = best_neuron_model.evaluate(X_train, y_train)
    print(f'Achieved training accuracy of {accuracy}')   
    _, accuracy = best_neuron_model.evaluate(X_val, y_val)
    print(f'Achieved validation accuracy of {accuracy}')
    _, accuracy = best_neuron_model.evaluate(X_test, y_test)
    print(f'Achieved test accuracy of {accuracy}')
    
  
    #test the best epoch model
    X_train, X_val = X_trainval[epoch_splits[0]], X_trainval[epoch_splits[1]]
    y_train, y_val = y_trainval[epoch_splits[0]], y_trainval[epoch_splits[1]]
    
    print(f'With epoch count of {best_epoch}:')
    _, accuracy = best_epoch_model.evaluate(X_train, y_train)
    print(f'Achieved training accuracy of {accuracy}')
    _, accuracy =  best_epoch_model.evaluate(X_val, y_val)
    print(f'Achieved validation accuracy of {accuracy}')
    _, accuracy =  best_epoch_model.evaluate(X_test, y_test)
    print(f'Achieved test accuracy of {accuracy}')
    
  train_validate_test_performance('best_layer_model.h5', 'best_neuron_model.h5', 'best_epoch_model.h5', layer_splits, neuron_splits, epoch_splits, best_layer, best_neuron, best_epoch)
    
    
thyroid_autoexperiments('Thyroid_Diff.csv')
  
  
  
# model, validation_performance = thyroid_cancer_recurrence_model('Thyroid_Diff.csv')

