
# Analiza botów z twittera

### 1. Zawartość kodów.

Kody podzielono na cztery pliki.
- data_prep.py zawiera pomocnicze funkcje do wstępnej obróbki zbioru danych
- convolutional.py - zawiera kod używany do badania efektywności algorytmu opartego na GloVe i sieciach splotowych
- tfidf.py - zawiera kod używany do badania efektywności algorytmów opartych o TF-IDF
- ft.py - zawiea kod używany do badania efektywności algorytmów opartych o fastText

### 2. Uruchomienie

Do uruchomienia poniższych plików wymagane jest stworzenie katalogu 'resources' w którym znajdą się:

- zbiór treningowy od Kaggle - train.csv (https://www.kaggle.com/competitions/twitter-spam/data)
- glove.6B.100d.txt 
- cc.en.300.bin (fastText)