from movies import movie_dataset, movie_ratings
from sklearn.neighbors import KNeighborsRegressor

regressor = KNeighborsRegressor(n_neighbors = 5, weights = "distance")

regressor.fit(movie_dataset, movie_ratings)
unknown_points = [[0.016, 0.300, 1.022],
[0.0004092981, 0.283, 1.0112],
[0.00687649, 0.235, 1.0112]]
print(regressor.predict(unknown_points))



