from sklearn.externals import joblib

clf2 = joblib.load('model.pkl')

X_test = [[7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]]
# 5

X_test2 = [[10.9, 0.53, 0.49, 4.6, 0.118, 10, 17, 1.0002, 3.07, 0.56, 11.7]]
# 6

# Predict data set using loaded model
print clf2.predict(X_test)
print clf2.predict(X_test2)