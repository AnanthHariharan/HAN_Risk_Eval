import pandas as pd
import numpy as np
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Flatten, Embedding, Dense, LSTM, GRU, Bidirectional, TimeDistributed, Concatenate, Activation, Multiply, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping


# Load CSV file into a pandas dataframe
data = pd.read_csv('g06_small.csv')
'''
# Preprocessing text data
def preprocess(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert text to lowercase
    text = text.lower()
    # Remove stopwords
    #stop_words = set(stopwords.words('english'))
    words = text.split()
    #words = [word for word in words if not word in stop_words]
    # Stem words
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    # Convert words back to string
    text = ' '.join(words)
    return text

'''
preprocessed_data = []
text_data = data['abstract']
for text in text_data:
    preprocessed_data.append(text)
'''
for text in text_data:
    preprocessed_text = preprocess(text)
    preprocessed_data.append(preprocessed_text)
'''
# Convert the labels to numerical form
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['Citation#'])

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(preprocessed_data, labels, test_size=0.5, random_state=42)

# Tokenize the text data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(preprocessed_data)
train_sequences = tokenizer.texts_to_sequences(train_data)
test_sequences = tokenizer.texts_to_sequences(test_data)

# Pad the sequences to a fixed length
max_seq_length = 100
train_sequences = pad_sequences(train_sequences, maxlen=max_seq_length, padding='post', truncating='post')
test_sequences = pad_sequences(test_sequences, maxlen=max_seq_length, padding='post', truncating='post')

# Define the HAN model architecture
input_layer = Input(shape=(max_seq_length,))
embedding_layer = Embedding(input_dim=5000, output_dim=100, input_length=max_seq_length)(input_layer)
word_encoder = Bidirectional(GRU(64, return_sequences=True))(embedding_layer)
word_attention = TimeDistributed(Dense(1, activation='tanh'))(word_encoder)
word_attention = Flatten()(word_attention)
word_attention = Activation('softmax')(word_attention)
word_representation = Multiply()([word_encoder, word_attention])
sentence_encoder = Bidirectional(GRU(64, return_sequences=True))(word_representation)
sentence_attention = TimeDistributed(Dense(1, activation='tanh'))(sentence_encoder)
sentence_attention = Flatten()(sentence_attention)
sentence_attention = Activation('softmax')(sentence_attention)
sentence_representation = Multiply()([sentence_encoder, sentence_attention])
word_level_output = TimeDistributed(Dense(1, activation='softmax'))(word_representation)
document_representation = Lambda(lambda x: np.mean(x, axis=1))(sentence_representation)
document_level_output = Dense(1, activation='sigmoid')(document_representation)
model = Model(inputs=input_layer, outputs=[document_level_output, word_level_output])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
history = model.fit(train_sequences, [train_labels, train_sequences], epochs=10, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_accuracy, _, _ = model.evaluate(test_sequences, [test_labels, test_sequences], verbose=0)
print('Test accuracy:', test_accuracy)

# Save the model
model.save('han_model.h5')