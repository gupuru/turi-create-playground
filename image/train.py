import turicreate as tc

data =  tc.SFrame('data/train.sframe')
train_data, test_data = data.random_split(0.8)

model = tc.image_classifier.create(train_data, target='label', max_iterations=100)

metrics = model.evaluate(test_data)
print(metrics['accuracy'])

model.save('data/dogs.model')
model.export_coreml('data/dogs.mlmodel')