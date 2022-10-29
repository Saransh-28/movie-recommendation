# Import the required libraries
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# load the dataset
df = pd.read_csv('movies.csv')

# features to be combined
features = ['genres','keywords','tagline','cast','director']

# replace all the null values in the selected features with the empty string
for feature in features:
  df[feature] = df[feature].fillna('')

# combine all the features
combined_features = df['genres']+' '+df['keywords']+' '+df['tagline']+' '+df['cast']+' '+df['director']

# vectorize the combined feature
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

def recommend():
    # input the movie name
    movie_name = input('Enter the movie name : ')

    # load the list of all the availabe movie
    list_movies = df['title'].tolist()

    # find the closest match of the input name
    name_movie = difflib.get_close_matches(movie_name, list_movies)[0]
    index_movie = df[df.title == name_movie]['index'].values[0]

    # get the similarity score for the given movie name
    similarity_score = list(enumerate(similarity[index_movie]))

    # sort the list of movies
    sorted_movies_list = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    # list top 20 recommended movies
    print('Movies suggested for you :')
    i = 1
    for movie in sorted_movies_list:
        index = movie[0]
        title_from_index = df[df.index == index]['title'].values[0]
        if (i < 20):
            print(f'{i:_^10}{title_from_index}')
            i += 1

recommend()