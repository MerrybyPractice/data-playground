import json
import numpy as np 
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json

tokenizer = Tokenizer()
wine_training = np.genfromtxt('ranked_wine.csv', delimiter=',', skip_header=1,usecols=(1,3),dtype=None, filling_values= 0, invalid_raise=False, encoding=None)
train_x = [x[0] for x in wine_training]
tokenizer.fit_on_texts(texts=train_x)

labels = ['positive', 'neutral', 'negative']

with open('dictionary.json', 'r') as dictionary_file: 
  dictionary = json.load(dictionary_file)

def convert_text_to_index_array(text): 
  words = kpt.text_to_word_sequence(text)
  wordIndices = []
  for word in words: 
    if word in dictionary: 
      wordIndices.append(dictionary[word])
    else: 
      print("'%s' not in training corpus; ignoring." %(word))
  return wordIndices

json_file = open('tipsy_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights('tipsy_model.h5')

while 1: 
  evalSentence = input('Input a sentence to be evaluated, or Enter to quit: ')

  if len(evalSentence) == 0: 
    break

  testArr = convert_text_to_index_array(evalSentence)
  input = tokenizer.sequences_to_matrix([testArr], mode='binary')
  # Takes my new string and scores it! 
  pred = model.predict(input)

  print(pred, "%s sentiment; %f%% confidence" % (labels[np.argmax(pred)], pred[0][np.argmax(pred)]*100))