# Data Science, NLP and Sentiment Classification

## Parts of analysis

* Tokenize: Break that string apart.
* Sequence Mapping: What is in this string?
* Neural Network:
  * one-hot vs vector embeddings

## Picking a data set

* Be aware of what you are training your bot to classify. A set of reviews of service will correlate high star ratings and positive words to good customer experiences (and potentially experiences in general) a set of wine reviews will correlate high star ratings - and therefore weights - with words used to describe good wine.

## Bits of the code

### Pipenv installs

* Pandas
* Keras
* tensorflow?

### Selecting for the text and the rating 

* The analysis we will be doing only requires the text and numeric rating from your data set. Shape the imported CSV accordingly. (In the case of this data set we could have used the price as well. What would that choice have implied?)

### Keras

* Tokenizer:
  
  keras.preprocessing.text.Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=False, oov_token=None, document_count=0)

  num_words can be used to set a max number if your data set is quite large and you only want to train on the most common words in the set.
  
  * 'ModuleNotFoundError: No module named tensorflow - ensure tensorflow has been installed.
  * 'Off by one' error in tonkenizer is feature

* Tokenizer.fit_on_texts()

  Fit on texts takes the tokenized string, tabulates the frequency of each token (word, char grouping, ect), and assigns an index. The lower the index, the more frequent the word is in the set.

* Tokenizer.texts_to_sequences()

  Assigns points based on the index's calculated via fit_on_texts.

* pad_sequences

  Pads sequences so that they are the same length. Set the final param, val, to what ever you want the max length of your sequences to be (ie, if you set val=100 and have a text sample of 101, are you ok with truncating that last value?)

  keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0)

* model 
<!-- TODO:Fill out -->

  .add 

* Dense()
<!-- TODO:Fill out -->
* to_categorical: keras.utils.to_categorical(y, num_classes=None, dtype='float32')

* Layers: 

embedding: 
SpatialDropout1D: 
Bidirectional: 
GlobalMaxPool1D: 
AttributeError: 'Tensor' object has no attribute 'lower' - Need to pass the layer through in this format: maxed = GlobalMaxPool1D()(rnn)
GlobalAvgPool1D: 
AttributeError: 'Tensor' object has no attribute 'lower' = avged = GlobalAvgPool1D()(rnn)
Concatenate: 
Dense:
Activation Functions: 
  Guide judgement on the fit of node weight - default is linear.
Dropout: 
  randomly drop data - avoids overfitting. Over fitting occurs when you train on data that is too similar and your accuracy holds steady or drops. We always want accuracy to rise.

* Training: 
  * model.fit
  * train_x and train_y: the input categories we are training on. In this case points(sentiment) and description(text, reviews, tweets, novels - what do you want it to be?)
  * Epochs: How many times?
  * validation_split: how much of the data is reserved for testing? typically 80:20 or 90:10

### Pandas

## Sources

https://keras.io/

https://colab.research.google.com/drive/1OlQpHdZD7zVyZW56r8vI-L8BYylq_Umm#scrollTo=Np5170sqkDjo

https://vgpena.github.io/classifying-tweets-with-keras-and-tensorflow/

https://stackoverflow.com/questions/51956000/what-does-keras-tokenizer-method-exactly-do

https://github.com/keras-team/keras/issues/7551

https://stackoverflow.com/questions/53153790/tensor-object-has-no-attribute-lower

