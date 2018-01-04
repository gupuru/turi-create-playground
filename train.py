import turicreate as tc
data = tc.SFrame.read_csv('data/ml-latest-small/ratings.csv')
data.explore()
training_data, validation_data = tc.recommender.util.random_split_by_user(data, 'userId', 'movieId')

print("#### train #####")
model = tc.recommender.create(training_data, 'userId', 'movieId')

results = model.recommend()