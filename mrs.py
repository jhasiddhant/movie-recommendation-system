import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords


# Access Datasets
credits_csv = pd.read_csv('./datasets/credits.csv')
keywords_csv = pd.read_csv('./datasets/keywords.csv')
movies_csv = pd.read_csv('./datasets/movies_metadata.csv', dtype='unicode')


# Data Cleaning
movies_csv = movies_csv[['id', 'original_title', 'overview', 'genres']]
movies_csv['title'] = movies_csv['original_title']
movies_csv.reset_index(inplace=True, drop=True)
movies_csv.drop('original_title', axis=1, inplace=True)

movies_csv['id'] = movies_csv['id'].astype(int)
dataset = pd.merge(movies_csv, keywords_csv, on='id', how='left')
dataset.reset_index(inplace=True, drop=True)
dataset = pd.merge(dataset, credits_csv, on='id', how='left')
dataset.reset_index(inplace=True, drop=True)

dataset['genre'] = dataset['genres'].apply(lambda x: [i['name'] for i in eval(x)])
dataset['genre'] = dataset['genre'].apply(lambda x: ' '.join([i.replace(" ", "") for i in x]))
dataset.drop('genres', axis=1, inplace=True)

dataset['keywords'].fillna('[]', inplace=True)
dataset['keywords'] = dataset['keywords'].apply(lambda x: [i['name'] for i in eval(x)])
dataset['keywords'] = dataset['keywords'].apply(lambda x: ' '.join([i.replace(" ", "") for i in x]))

dataset['cast'].fillna('[]', inplace=True)
dataset['cast'] = dataset['cast'].apply(lambda x: [i['name'] for i in eval(x)])
dataset['cast'] = dataset['cast'].apply(lambda x: ' '.join([i.replace(" ", "") for i in x]))

dataset['crew'].fillna('[]', inplace=True)
dataset['crew'] = dataset['crew'].apply(lambda x: [i['name'] for i in eval(x)])
dataset['crew'] = dataset['crew'].apply(lambda x: ' '.join([i.replace(" ", "") for i in x]))

dataset['tags'] = dataset['overview']+' '+dataset['keywords']+' '+dataset['cast']+' '+dataset['crew']+' '+dataset['genre']+' '+dataset['title']

dataset.drop(['genre','keywords', 'cast', 'crew', 'overview'], axis=1, inplace=True)


# Vectorizing data
sw_nltk = stopwords.words('english')
vectorizer = TfidfVectorizer(max_features=5000, stop_words=sw_nltk)

vectorized_data = vectorizer.fit_transform(dataset['tags'].values.astype('U'))
vectorizer.get_feature_names_out()

vectorized_dataframe = pd.DataFrame(vectorized_data.toarray(), index=dataset['tags'].index.tolist())
vectorized_dataframe.shape


# Dimentionality Reduction
svd = TruncatedSVD(n_components=1000)

reduced_data = svd.fit_transform(vectorized_dataframe)
reduced_data.shape

svd.explained_variance_ratio_.cumsum()


# Making Recommendations
reduced_data = reduced_data.astype(np.float32)
similarity = cosine_similarity(reduced_data)


def recomendations(movie):
    id_of_movie = dataset[dataset['title']==movie].index[0]
    distances = similarity[id_of_movie]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:15]
    for movie_id in movie_list:
        print(dataset.iloc[movie_id[0]].title)


recomendations('Avengers: Age of Ultron')