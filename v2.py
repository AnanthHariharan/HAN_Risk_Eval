import pandas as pd
import numpy as np
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load the CSV file into a pandas dataframe
data = pd.read_csv('g06_small.csv')

# Preprocess the text data
def preprocess(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert text to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if not word in stop_words]
    # Stem words
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    # Convert words back to string
    text = ' '.join(words)
    return text

text_data = data['abstract']
preprocessed_data = []
for text in text_data:
    preprocessed_text = preprocess(text)
    preprocessed_data.append(preprocessed_text)

# Tokenize the text data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(preprocessed_data)
sequences = tokenizer.texts_to_sequences(preprocessed_data)
max_seq_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post', truncating='post')

# Load the saved HAN model
model = load_model('han_model.h5')

# Use the model to predict the importance of each sentence in the text data
document_weights, sentence_weights = model.predict(padded_sequences)

# Write the importance values for each sentence in the document to a file
with open('output.txt', 'w') as f:
    for j, doc in enumerate(text_data):
        f.write('Importance values for each sentence in document {}:\n'.format(j+1))
        for i, sentence in enumerate(doc.split('. ')):
            row_number = data.index[j] + 1
            f.write('Row Number: {}, Sentence {}: {}\n'.format(row_number, i+1, sentence))
            f.write('Sentence weight: {}\n\n'.format(sentence_weights[j][i]))
