import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from yellowbrick.text import UMAPVisualizer

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def prepare(only_nlp, path):
    df = pd.read_csv(path, usecols=['Tweet', 'following', 'followers', 'actions',
                                    'is_retweet', 'location', 'Type'])
    df.drop(df.filter(regex="Unnamed"), axis=1, inplace=True)

    if only_nlp:
        df.drop(['following', 'followers', 'actions', 'is_retweet', 'location'], axis=1, inplace=True)

    df.drop(df.index[(df["Type"] == 'South Dakota')], axis=0, inplace=True)
    df.reset_index(drop=True)
    return df


df_train = prepare(True, 'resources/train.csv')
df_train.to_csv('resources/train_afterprep.csv', index=False)
df = pd.read_csv('resources/train_afterprep.csv')
df['split'] = np.random.randn(df.shape[0], 1)
msk = np.random.rand(len(df)) <= 0.7

train = df[msk]
test = df[~msk]
train.to_csv('resources/train2.csv', index=False)
test.to_csv('resources/test2.csv', index=False)

test_csv = pd.read_csv('resources/test2.csv')
train_csv = pd.read_csv('resources/train2.csv')

nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()

train_X_non = train_csv['Tweet']  # '0' refers to the review text
train_y = train_csv['Type'].map({'Spam': 1, 'Quality': 0})  # '1' corresponds to Label (1 - positive and 0 - negative)
test_X_non = test_csv['Tweet']
test_y = test_csv['Type'].map({'Spam': 1, 'Quality': 0})
train_X = train_X_non
test_X = test_X_non

for i in range(0, len(train_X_non)):
    review = review.sub('[^a-zA-Z]', ' ', train_X_non[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
    review = ' '.join(review)
    train_X.append(review)

# text pre processing
for i in range(0, len(test_X_non)):
    review = review.sub('[^a-zA-Z]', ' ', test_X_non[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
    review = ' '.join(review)
    test_X.append(review)

tf_idf = TfidfVectorizer()
X_train_tf = tf_idf.fit_transform(train_X)
X_test_tf = tf_idf.transform(test_X)

clusters = KMeans(n_clusters=5)
clusters.fit(X_train_tf)

umap = UMAPVisualizer()
umap.fit(X_train_tf, ["c{}".format(c) for c in clusters.labels_])
umap.show()

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_tf.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_tf.toarray(), train_y, validation_data=(X_test_tf.toarray(), test_y), batch_size=64, epochs=10)

predictions = model.predict(X_test_tf.toarray())
y_pred = [1 if x > 0.5 else 0 for x in predictions]
matrix = confusion_matrix(test_y, y_pred)

print(metrics.classification_report(test_y, y_pred, target_names=['Spam', 'Quality']))
print(matrix)
