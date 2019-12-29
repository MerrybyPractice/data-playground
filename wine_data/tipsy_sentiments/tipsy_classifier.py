import numpy as np
# ML imports
# Data Imports
import json
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
#model imports
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


# Pull and clean data to feed model

wine_training = np.genfromtxt('ranked_wine.csv', delimiter=',', skip_header=1,usecols=(1,3),dtype=None, filling_values= 0, invalid_raise=False, encoding=None)

# pull off description and create train_y data from it.
train_x = [x[0] for x in wine_training]
# print(train_x)
# pull off sentiment and create train_x data from it.
train_y = np.asarray([x[1] for x in wine_training])

# print(train_y)

tokenizer = Tokenizer()
#tokenize dat wine!
tokenizer.fit_on_texts(texts=train_x)

dictionary = tokenizer.word_index

with open('dictionary.json', 'w') as dictionary_file:
  json.dump(dictionary, dictionary_file)

def convert_text_to_index_array(text):
  return [dictionary[word] for word in kpt.text_to_word_sequence(text)]

allWordIndices = []

for text in train_x: 
  wordIndices = convert_text_to_index_array(text)
  allWordIndices.append(wordIndices)

for string in train_y: 
  string = int(string)

allWordIndices = np.asarray(allWordIndices)

# one-hot matricize
train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary') 

train_y = train_y -1
# print(train_y)
train_y = keras.utils.to_categorical(train_y, 3)

# Creating Model
# layers will be exectued in order, treated like stack
model = Sequential()
# Layer 1, dropout 1
# 512 outputs from layer 1, input_shape could take max words, if necessary, activation function relu
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
# Layer 2, dropout 2
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.5))
# Layer 3
model.add(Dense(3, activation='softmax'))

#Compile the drinkers (i mean network, not a lot of virtual people hanging out and rating wine, that would be absurd)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training comenses 

model.fit(train_x, train_y, 
batch_size=32,
epochs=5,
verbose=1,
validation_split=0.1,
shuffle=True)

model_json = model.to_json()
with open('tipsy_model.json', 'w') as json_file: 
  json_file.write(model_json)

model.save_weights('tipsy_model.h5')
