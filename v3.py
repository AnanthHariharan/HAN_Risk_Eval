import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Dense, Bidirectional, GRU, TimeDistributed, Flatten, concatenate, Permute, \
    Reshape
from keras.models import Model
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
import keras.backend as K


df = pd.read_csv('us_accidents.csv')

# Preprocess the text data
tokenise = Tokenizer(num_words=10000)
tokenise.fit_on_texts(df['Description'])
sequences = tokenise.texts_to_sequences(df['Description'])
X = pad_sequences(sequences, maxlen=100)

# Prepare the target variable
y = df['Severity'].values

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

# HAN model architecture
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


word_input = Input(shape=(100,), dtype='int32')
word_embedding = Embedding(10000, 128, input_length=100)(word_input)
word_bidir = Bidirectional(GRU(64, return_sequences=True))(word_embedding)
word_att = Attention()(word_bidir)
word_encoder = Model(word_input, word_att)

sent_input = Input(shape=(10, 100), dtype='int32')
sent_encoder = TimeDistributed(word_encoder)(sent_input)
sent_bidir = Bidirectional(GRU(64, return_sequences=True))(sent_encoder)
sent_att = Attention()(sent_bidir)

denser = Dense(64, activation='relu')(sent_att)
output = Dense(1, activation='sigmoid')(denser)

model = Model(inputs=sent_input, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
X_3d = np.array(np.split(X, 10))
model.fit(X_3d, y, batch_size=32, epochs=10, validation_split=0.2)

# Extracting weights of the attention layer
weights = model.layers[3].get_weights()[0]

# Identifying the most important risk predictors
word_index = tokenise.word_input
keywords = ['speed', 'weather', 'road']
predictors = []
for keyword in keywords:
    if keyword in word_index:
        index = word_index[keyword]
        predictor = weights[:, index]
        predictors.append(predictor)
predictors = np.concatenate(predictors, axis=1)
importance = np.sum(np.abs(predictors), axis=1)

# Print predictors (temp)
top_predictors = np.argsort(importance)[::-1][:10]
for i in top_predictors:
    predictor_words = [word for word, index in word_index.items() if index == i]
    print("Predictor words:", predictor_words)
'''
def evaluate_risk(model, text):
    # Preprocess the text data
    #preprocessed_text = preprocess(text)
    preprocessed_text = text
    # Tokenize and pad the sequence
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts([preprocessed_text])
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
    # Use the model to predict the risk level
    document_level_output, word_level_output = model.predict(sequence)
    risk_level = document_level_output[0][0]
    return risk_level
'''