from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,roc_auc_score,log_loss,precision_score,recall_score,f1_score
import pandas as pd
data = pd.read_csv("data.csv",index_col=False)
print(data.columns)
model = RandomForestClassifier(random_state=42,n_estimators=500)
X = data
X = X.drop(columns=['R'])
Y = data['R']
print(X,Y)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
model.fit(x_train,y_train)
pred = model.predict(x_test)
accS = accuracy_score(y_test,pred)
print("ACCS: " + str(accS))
