from model_new import NewModel

obj = NewModel()
ans = obj.predict_in_class('sample_data_new/11.jpg', 'is this a car')
print(ans)