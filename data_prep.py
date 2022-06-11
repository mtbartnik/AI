import pandas as pd
import string
from nltk.corpus import stopwords

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def prepare(only_nlp, path):
    df = pd.read_csv(path, usecols=['Tweet', 'following', 'followers', 'actions',
                                                                    'is_retweet', 'location', 'Type'])
    df.drop(df.filter(regex="Unnamed"), axis=1, inplace=True)

    if only_nlp:
        df.drop(['following', 'followers', 'actions', 'is_retweet', 'location'], axis=1, inplace=True)

    df['Type'] = df['Type'].map({'Spam': 1, 'Quality': 0})
    df.reset_index(drop=True)
    return df


