import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")


# Access Datasets
credits_csv = pd.read_csv('./datasets/credits.csv')
keywords_csv = pd.read_csv('./datasets/keywords.csv')
movies_csv = pd.read_csv('./datasets/movies_metadata.csv', dtype='unicode')


# Preprocess the datasets
movies_csv = movies_csv[['id', 'original_title', 'overview', 'genres']]
movies_csv['title'] = movies_csv['original_title']
movies_csv.reset_index(inplace=True, drop=True)
movies_csv.drop('original_title', axis=1, inplace=True)
movies_csv['title'] = movies_csv['title'].apply(str.upper)

movies_csv['id'] = movies_csv['id'].astype(int)
dataset = pd.merge(movies_csv, keywords_csv, on='id', how='left')
dataset.reset_index(inplace=True, drop=True)
dataset = pd.merge(dataset, credits_csv, on='id', how='left')
dataset.reset_index(inplace=True, drop=True)

def preprocess_column(df, column):
    df[column].fillna('[]', inplace=True)
    df[column] = df[column].apply(lambda x: [i['name'] for i in eval(x)])
    df[column] = df[column].apply(lambda x: ' '.join([i.replace(" ", "") for i in x]))
    return df

dataset = preprocess_column(dataset, 'genres')
dataset = preprocess_column(dataset, 'keywords')
dataset = preprocess_column(dataset, 'cast')
dataset = preprocess_column(dataset, 'crew')

# Create a new column 'tags' containing the overview, keywords, cast, crew and genres of the movie
dataset['tags'] = dataset['overview']+' '+dataset['keywords']+' '+dataset['cast']+' '+dataset['crew']+' '+dataset['genres']+' '+dataset['title']

# Drop the unnecessary columns
dataset.drop(['genres','keywords', 'cast', 'crew', 'overview'], axis=1, inplace=True)

# Create a set of stopwords
stop_words = set(stopwords.words("english"))

# Initialize the stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_tags(tags):
    if type(tags) == str:
        # Convert the tags to lowercase
        tags = tags.lower()

        # Remove punctuation
        tags = tags.translate(str.maketrans("", "", string.punctuation))

        # Tokenize the tags into words
        words = nltk.word_tokenize(tags)

        # Remove stopwords and stem/lemmatize the remaining words
        words = [lemmatizer.lemmatize(stemmer.stem(word)) for word in words if word not in stop_words]

        # Join the processed words back into a string
        return " ".join(words)
    else:
        return tags

# Apply the preprocessing function to the dataset['tags'] column
dataset['tags'] = dataset['tags'].apply(preprocess_tags)


''' Content Based Movie Recommender System '''

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
vectorized_data = vectorizer.fit_transform(dataset['tags'].values.astype('U'))
vectorized_dataframe = pd.DataFrame(vectorized_data.toarray(), index=dataset['tags'].index.tolist())


# Reduce the dimensionality of the vectorized data
svd = TruncatedSVD(n_components=1000)
reduced_data = svd.fit_transform(vectorized_dataframe)


# Calculate the cosine similarity between the movies
reduced_data = reduced_data.astype(np.float32)
similarity = cosine_similarity(reduced_data)


# Function to recommend movies
def recomendations(movie):
    # Getting the index of the movie
    if len(dataset[dataset['title']==movie].index) == 0:
        print("Movie not found in dataset")
        return
        
    id_of_movie = dataset[dataset['title']==movie].index[0]
    # Calculating cosine similarity between the movie and all other movies
    distances = similarity[id_of_movie]
    
    # Sorting the movies based on cosine similarity
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:11]
    
    # Printing the recommended movies
    for movie_id in movie_list:
        print(dataset.iloc[movie_id[0]].title)


# Taking user input for movie name
movie_name = input("Enter movie name: ")

# Converting movie name to uppercase
x = movie_name.upper()

# Calling the recommendation function
print("Recommended movies:\n",recomendations(x))