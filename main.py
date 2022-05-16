from keras.layers import BatchNormalization, Flatten
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras import layers
import data_prep
from sklearn.model_selection import train_test_split
import tensorflow as tf
from model import create_embedding_matrix

maxlen = 280
optimizer = tf.keras.optimizers.Adam(lr=0.001, clipnorm=0.1)
only_nlp = True

df_train = data_prep.prepare(only_nlp, 'resources/train.csv')
sentences = df_train['Tweet'].values
y = df_train['Type'].values

sentences_train, sentences_test, Y_train, Y_test = train_test_split(sentences, y, test_size=0.15, random_state=1000)

tokenizer = Tokenizer(num_words=36000)
tokenizer.fit_on_texts(sentences_train)
print('Found %d unique words.' % len(tokenizer.word_index))
X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

vocab_size = len(tokenizer.word_index) + 1

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

embedding_dim = 100
embedding_matrix = create_embedding_matrix('resources/glove.6B.100d.txt', tokenizer.word_index, embedding_dim)

model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
model.add(layers.Conv1D(256, 5, activation='relu'))
model.add(BatchNormalization())
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(BatchNormalization())
model.add(layers.Dense(1, activation='sigmoid'))
model.add(BatchNormalization())
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    epochs=10,
                    validation_data=(X_test, Y_test),
                    batch_size=10)

