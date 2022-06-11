from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras import layers
import data_prep
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

batch_size = 32
embedding_dims = 300
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2
maxlen = 100
optimizer = tf.keras.optimizers.Adam(lr=0.001, clipnorm=0.1)
only_nlp = True

df_train = data_prep.prepare(only_nlp, 'resources/train.csv')
sentences = df_train['Tweet'].values
y = df_train['Type'].values

sentences_train, sentences_test, Y_train, Y_test = train_test_split(
    sentences, y,
    test_size=0.15,
    random_state=1000)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

vocab_size = len(tokenizer.word_index) + 1

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1
    # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


embedding_dim = 50
embedding_matrix = create_embedding_matrix('resources/glove.6B.50d.txt',
                                           tokenizer.word_index,
                                           embedding_dim)

embedding_dim = 100

model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(BatchNormalization())
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(5, activation='relu'))
model.add(BatchNormalization())
model.add(layers.Dense(1, activation='relu'))
model.add(BatchNormalization())
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    epochs=10,
                    validation_data=(X_test, Y_test),
                    batch_size=10)