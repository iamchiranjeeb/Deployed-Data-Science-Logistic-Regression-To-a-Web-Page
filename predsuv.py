import pandas as pd
from pandas_ods_reader import read_ods
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import warnings

warnings.filterwarnings("ignore")

sheetidx = 1
path = "/home/iamchiranjeeb/Desktop/Python/suv.ods"
df = read_ods(path,sheetidx)

df = df.drop(['User_id'],axis=1)
pd.get_dummies(df["Gender"])
sex = pd.get_dummies(df["Gender"],drop_first=True)

df = pd.concat([df,sex],axis=1)
df = df.drop(['Gender'],axis=1)
df = df[['Age','Est_salary','male','Purchased']]

X = df.iloc[:,:3]
y = df.iloc[:,-1:]

X = X.astype('int')
y = y.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=0)

reg = LogisticRegression(solver='lbfgs')
reg.fit(X_train,y_train)

inputt=[int(x) for x in "45 60000 0".split(' ')]
final=[np.array(inputt)]

# yp = reg.predict(X_test)

yp2 = reg.predict(final)
b = reg.predict_proba(final)

pickle.dump(reg,open('suv.pkl','wb'))
model=pickle.load(open('suv.pkl','rb'))