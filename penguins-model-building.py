import pandas as pd
penguins = pd.read_csv('dataset_train2.csv')
from sklearn.model_selection import train_test_split
from dash import html

# Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = penguins.copy()
target = 'TARGET'



# Separating X and y
Y = df["TARGET"]
X = df.loc[:, df.columns != 'TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.6, stratify = Y, random_state=0)

# Build random forest model
from xgboost import XGBClassifier
XG=XGBClassifier()
XG.fit(X_train , y_train)

from explainerdashboard import ClassifierExplainer, ExplainerDashboard
explainer = ClassifierExplainer(XG, X_test, y_test)

ExplainerDashboard(explainer).run()

# Saving the model
import pickle
pickle.dump(XG, open('penguins_clf.pkl', 'wb'))
