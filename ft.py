import numpy as np
import pandas as pd
import fasttext

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

df_train['split'] = np.random.randn(df_train.shape[0], 1)

msk = np.random.rand(len(df_train)) <= 0.9
train = df_train[msk]
test = df_train[~msk]

train.iloc[:, 1] = train.iloc[:, 1].apply(lambda x: '__label__' + x)
train.to_csv('resources/ft.txt', index=False, sep=' ', header=None)

test.iloc[:, 1] = test.iloc[:, 1].apply(lambda x: '__label__' + x)
test.to_csv('resources/ft_test.txt', index=False, sep=' ', header=None)

model = fasttext.train_supervised("resources/ft.txt")
print(model.labels)


def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))


print_results(*model.test('resources/ft_test.txt'))
