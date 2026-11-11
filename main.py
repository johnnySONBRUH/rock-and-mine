from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import joblib
import shap
data = pd.read_csv("data.csv",index_col=False)
model = RandomForestClassifier(random_state=42,n_estimators=500,max_depth=10)
X = data
X = X.drop(columns=['R'])
Y = data['R']
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
model.fit(x_train,y_train)
pred = model.predict(x_test)
acc = accuracy_score(y_test,pred)
prec = precision_score(y_test,pred,average='macro')
f1 = f1_score(y_test,pred,average='macro')
rec = recall_score(y_test,pred,average='macro')
print("ACCURACY SCORE: " + str(acc))
print("PRECISION SCORE: " + str(prec))
print("RECALL SCORE: " + str(rec))
print("F1 SCORE: " + str(f1))
joblib.dump(model,'prediction.pkl')
