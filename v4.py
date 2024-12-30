import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, GRU, Bidirectional, TimeDistributed, Layer, Embedding
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping

class AttentionLayer(Layer):
    def __init__(self, name, **kwargs):
        super(AttentionLayer, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],), initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        q = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(K.sum(q, axis=-1))
        return K.sum(x * K.expand_dims(a), axis=1)


def create_han_model(vocab_size, max_sentences, max_words, embedding_dim):
    word_input = Input(shape=(max_words,), dtype='int32')
    word_sequences = Embedding(vocab_size, embedding_dim)(word_input)
    word_gru = Bidirectional(GRU(embedding_dim, return_sequences=True))(word_sequences)
    word_attention = AttentionLayer(name="word_attention")(word_gru)
    word_encoder = Model(word_input, word_attention)

    sentence_input = Input(shape=(max_sentences, max_words), dtype='int32')
    sentence_sequences = TimeDistributed(word_encoder)(sentence_input)
    sentence_gru = Bidirectional(GRU(embedding_dim, return_sequences=True))(sentence_sequences)
    sentence_attention = AttentionLayer(name="sentence_attention")(sentence_gru)
    output = Dense(1, activation='linear')(sentence_attention)

    model = Model(sentence_input, output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model

df = pd.read_csv('g06_small.csv')
df = df.dropna(subset=['abstract'])
texts = df['abstract'].tolist()
weights = df['Citation#'].tolist()

max_words = 50
max_sentences = 10
vocab_size = 10000
embedding_dim = 100

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index

data = np.zeros((len(texts), max_sentences, max_words), dtype='int32')
for i, text in enumerate(texts):
    sentences = text.split('.')
    for j, sentence in enumerate(sentences):
        if j < max_sentences:
            tokens = tokenizer.texts_to_sequences([sentence])[0]
            tokens = tokens[:max_words]
            data[i, j, :len(tokens)] = tokens

X_train, X_test, y_train, y_test = train_test_split(data, np.array(weights), test_size=0.2, random_state=42)


model = create_han_model(vocab_size, max_sentences, max_words, embedding_dim)
model.summary()

epochs = 10
batch_size = 32
patience = 3

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
predictions = model.predict(X_test)

def get_attention_weights(model, layer_name):
    for layer in model.layers:
        if isinstance(layer, TimeDistributed):
            for sublayer in layer.layer.layers:
                if sublayer.name == layer_name:
                    attention_layer = sublayer
                    break
            else:
                continue
            break
    else:
        raise ValueError(f"Layer with name {layer_name} not found.")

    #new input for the attention model
    attention_input = Input(shape=(max_words, ), dtype='int32')

    # word encoder layers to process the new input (reused)
    attention_output = attention_layer(layer.layer(attention_input))

    # attention model
    attention_model = Model(inputs=attention_input, outputs=attention_output)
    return attention_model

# attention weights for words
attention_model = get_attention_weights(model, "word_attention")

# attention weights for each sentence in the test set
word_attention_weights = []
for sentence_matrix in X_test:
    sentence_weights = []
    for sent in sentence_matrix:
        sent = np.expand_dims(sent, axis=0)
        weights = attention_model.predict(sent)
        sentence_weights.append(weights[0])
    word_attention_weights.append(sentence_weights)
word_attention_weights = np.array(word_attention_weights)

# Sentences, predictions, and the most important predictor words
with open('results.txt', 'w') as f, open('results.txt', 'w') as g:
    for i, prediction in enumerate(predictions):
        sentence_matrix = X_test[i]
        sentences = []
        for sent, word_weights in zip(sentence_matrix, word_attention_weights[i]):
            sentence = ' '.join([tokenizer.index_word[word_idx] if word_idx in tokenizer.index_word else '' for word_idx in sent])
            sentences.append(sentence.strip())
            sorted_word_indices = np.argsort(word_weights)[::-1]
            important_words = [tokenizer.index_word[sent[word_idx]] for word_idx in sorted_word_indices[:5] if sent[word_idx] in tokenizer.index_word]

            g.write(f"Abstract: {sentence.strip()}\n")
            g.write(f"Important predictor words: {', '.join(important_words)}\n")
            g.write(f"Text: {text}\n")
            g.write(f"Predicted citation: {prediction[0]:.2f}\n, Actual citation: {y_test[i]}\n")
            g.write("\n")
            g.write("\n")

        text = ' '.join(sentences).strip()

        
