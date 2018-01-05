import turicreate as tc
model = tc.load_model('model/recommend.model')
movie = tc.SFrame.read_csv('data/ml-latest-small/movies.csv')

print("=== input ===")
data = tc.SFrame({
    'userId': [3,15,128],
    'movieId': [1,6,111]
    })
print(data.join(movie, on='movieId'))

print("=== recommend ===")
res = model.recommend([3, 15, 128], new_observation_data=data)
print(res.join(movie, on='movieId').sort('rank'))

print("=== similar ===")
similar = tc.SFrame({'movieId': [6]})
print(similar.join(movie, on='movieId'))
print(model.get_similar_items(similar['movieId']).join(movie, on={'similar':'movieId'}).sort('score', ascending = False))
