import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("Salary_dataset.csv")
X = df[["YearsExperience"]]
Y = df[["Salary"]]

X.head()


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.60, random_state=48
)

model = LinearRegression() 
model.fit(X_train, Y_train)

s = model.predict(X_test)
print(s)
score = r2_score(s, Y_test)
print(score)

print(model.coef_)
print(model.intercept_)

plt.plot(X, Y, "o")
plt.plot(X, model.predict(X), color="red")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

result = model.predict([[8]])
result

import pickle 
pickle.dump(model, open("api_salary.pkl", "wb"))
