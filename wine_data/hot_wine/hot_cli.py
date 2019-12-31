import json
import numpy as np 
import tensorflow.keras
import tensorflow.keras.preprocessing.text as kpt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import model_from_json

#Pull in my CSV, create a DF with just my text and my sentiment
hot_wine_df = np.genfromtxt('ranked_wine.csv', delimiter=',', skip_header=1, usecols=(1,3), dtype=None, filling_values=0, invalid_raise=False, encoding=None)
#store the texts in hot_wine_x
hot_wine_x = [x[0] for x in hot_wine_df]

#*jazzhands* Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts=hot_wine_x)

#create your dict from json
with open('hot_wine_dict.json', 'r') as d: 
    hot_wine_dict = json.load(d)

#create your model from json
hot_wine_file = open('hot_wine_model.json', 'r')
hot_wine_loaded = hot_wine_file.read()
hot_wine_file.close()
hot_wine_model = model_from_json(hot_wine_loaded)

#remember that wild .h5 file? Now we load it. 
hot_wine_model.load_weights('hot_wine_model.h5')

#create an array of labels
sentiment_labels = ['Positive', 'Neutral', 'Negative']

def convert_text_to_index_array(text):
    words = kpt.text_to_word_sequence(text)
    word_idx = []
    for word in words: 
        if word in hot_wine_dict: 
            word_idx.append(hot_wine_dict[word])
        else: 
            print("I dont know '%s', skipping. Asking a they for an opinion with out preperation, the nerve." %(word))
    return word_idx

#now the CLI part
while 1: 
    wine_talk = input('What is it you wanted my opinion on? If your just going to wast my time, hit Enter to quit. ')

    if len(wine_talk) == 0: 
        break

    chat_array = convert_text_to_index_array(wine_talk)
    input = tokenizer.sequences_to_matrix([chat_array])
    #get that bots opinion 
    pred = hot_wine_model.predict(input)

    print(pred, "%s sentiment; %f%% confidence" %(sentiment_labels[np.argmax(pred)], pred[0][np.argmax(pred)]*100))
    
