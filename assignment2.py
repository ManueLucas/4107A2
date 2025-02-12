# Name this file assignment2.py when you submit
import numpy as np
import keras
from keras import Sequential, Model, layers, optimizers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
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

def thyroid_model_FFN():
  inputs = layers.Input(shape=(55,))
  
  hidden1 = layers.Dense(128, activation='relu')(inputs)
  hidden2 = layers.Dense(32, activation='relu')(hidden1)
  hidden3 = layers.Dense(8, activation='relu')(hidden2)
  hidden4 = layers.Dense(2, activation='relu')(hidden3)
  
  output = layers.Dense(1, activation='sigmoid')(hidden4)
  
  model = Model(inputs=inputs, outputs=output)
  model.compile(optimizer=optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
  
  return model
  
# A function that creates a keras model to predict whether a patient has recurrence of thryroid cancer
def thyroid_cancer_recurrence_model(filepath):
  # filepath is the path to an csv file containing the dataset
  data = pd.read_csv(filepath)  
  # for col in data:
  #   print(f'Unique values for column "{col}": {data[col].unique()}')
  data['Recurred'] = data['Recurred'].apply(lambda x: 1 if x == 'Yes' else 0)
  data = pd.get_dummies(data, columns=data.columns[1:-1])
  data_matrix = data.to_numpy()
  raw_y = data_matrix[:, 1].astype('float32')
  raw_X = np.delete(data_matrix, 1, axis=1).astype('float32')
  
  val_ratio = 0.2
  test_ratio = 0.2
  
  best_accuracy = 0
  best_model = None
  kf = KFold(n_splits=5, shuffle=True, random_state=42)
  
  for train_index, val_index in kf.split(raw_X):
    X_train, X_val = raw_X[train_index], raw_X[val_index]
    y_train, y_val = raw_y[train_index], raw_y[val_index]
    
    # Split test set from the training set
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_ratio, random_state=42)
    age_mean = np.mean(X_train[0], axis=0) # make sure we standardize using X_train to prevent data leakage
    age_std = np.std(X_train[0], axis=0)
    
    X_train[0] = (X_train[0] - age_mean) / age_std #standardize and normalize age since it's the only numerical feature
    X_val[0] = (X_val[0] - age_mean) / age_std
    X_test[0] = (X_test[0] - age_mean) / age_std
    
    model = thyroid_model_FFN()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
    validation_accuracy = history.history['val_accuracy'][-1]  # last validation accuracy
    if validation_accuracy > best_accuracy:
      best_accuracy = validation_accuracy
      best_model = model

  # model is a trained keras model for predicting whether a a patient has recurrence of thryroid cancer during a follow-up period
  # validation_performance is the performance of the model on a validation set

  return best_model, best_accuracy

model, validation_performance = thyroid_cancer_recurrence_model('Thyroid_Diff.csv')

print(validation_performance)