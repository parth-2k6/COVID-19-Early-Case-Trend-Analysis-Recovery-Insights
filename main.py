import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
df=pd.read_csv("patient.csv")
df['confirmed_date']=pd.to_datetime(df['confirmed_date'])
df['released_date']=pd.to_datetime(df['released_date'])
df['recovery_days']=(df['released_date'] - df['confirmed_date']).dt.days
df['age']=2020 - df['birth_year']
df=df.dropna(subset=['recovery_days', 'age', 'contact_number'])
print(df.describe())
# Visualization of the dataset
sns.histplot(df['age'], bins=20)
plt.savefig("outputs/age.png")
plt.clf()
sns.histplot(df['recovery_days'], bins=20)
plt.savefig("outputs/recovery.png")
plt.clf()
# ML Model for prediciton and R square value
X=df[['age', 'contact_number']]
y=df['recovery_days']
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2)
model=LinearRegression()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))