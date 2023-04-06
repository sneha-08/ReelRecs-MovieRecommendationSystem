import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv('/content/movies.csv',nrows=10000)
#ratings = pd.read_csv('/content/ratings.csv')
ratings = pd.read_csv("/content/ratings.csv", sep='\t',nrows=10000)

# Compute mean rating for each movie
movie_means = ratings.groupby('movieId')['rating'].mean().reset_index()
#movie_means = ratings.mean().reset_index()
movie_means.columns = ['movieId', 'mean_rating']

# Merge movie means with movie data
movies = pd.merge(movies, movie_means, on='movieId', how='left')
movies.head()

# Compute user-item matrix
ratings_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

# Replace missing values with movie means
ratings_matrix = ratings_matrix.fillna(movie_means['mean_rating'])

# Compute item-item similarity matrix using cosine similarity
item_sim_matrix = cosine_similarity(ratings_matrix.T)

# Recommend movies for a user based on similar items they rated highly
def recommend_movies(userId, num_recs=10):
    user_ratings = ratings_matrix.loc[userId]
    similar_items = pd.Series()
    for i, rating in user_ratings.iteritems():
        similar_items = similar_items.append(item_sim_matrix[i].\
                            apply(lambda x: (i, x*rating)))
    similar_items = similar_items.sort_values(ascending=False)
    similar_items = similar_items[~similar_items.index.isin(user_ratings.index)]
    recommendations = similar_items.groupby(level=0).apply(lambda x: x.\
                          nlargest(n=num_recs, columns=0)).reset_index(drop=True)
    recommendations = recommendations.apply(lambda x: (movies.loc[x[0]]['title'], x[1]), axis=1)
    return recommendations

# Example usage: recommend movies for user with userId=1
recommendations = recommend_movies(1, num_recs=10)
print(recommendations)

# Visualize movie means distribution
sns.displot(movies['mean_rating'], kde=False)
plt.xlabel('Mean Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Mean Ratings')
plt.show()

# Visualize item-item similarity matrix
sns.heatmap(item_sim_matrix, cmap='coolwarm')
plt.xlabel('Movie IDs')
plt.ylabel('Movie IDs')
plt.title('Item-Item Similarity Matrix')
plt.show()
