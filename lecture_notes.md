# Data Science, NLP and Sentiment Classification

## Parts of analysis

* Tokenize: Break that string apart.
* Sequence Mapping: What is in this string?
* Neural Network:

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
  
  * 'ModuleNotFoundError: No module named tensorflow - ensure tensorflow has been installed.

* Tokenizer.fit_on_texts()

  Fit on texts takes the tokenized string, tabulates the frequency of each token (word, char grouping, ect), and assigns an index. The lower the index, the more frequent the word is in the set.

* Tokenizer.texts_to_sequences()

  Assigns points based on the    

* pad_sequences

### Pandas

## Sources

https://keras.io/

https://colab.research.google.com/drive/1OlQpHdZD7zVyZW56r8vI-L8BYylq_Umm#scrollTo=Np5170sqkDjo
