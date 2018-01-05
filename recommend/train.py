import turicreate as tc
data = tc.SFrame.read_csv('../data/ml-latest-small/ratings.csv')
data.explore()
training_data, validation_data = tc.recommender.util.random_split_by_user(data, 'userId', 'movieId')

print("#### train #####")
model = tc.recommender.create(training_data, 'userId', 'movieId')

print("#### eval #####")
res = model.evaluate(validation_data)

pro = res['precision_recall_overall']
pro.print_rows(18,3)
tc.show(pro['recall'], pro['precision'], 'recall', 'precision')

model.save('../model/recommend.model')