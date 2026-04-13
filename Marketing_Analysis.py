import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, classification_report 

df = pd.read_csv(r"E:\Data Analysis Course\MACHINE LEARNING\MACHINE LEARNING REAL TIME PROJECT\Marketing Data Analysis\bank.csv", sep=';')
# print(df)

def balanceator(x, q99):
    if x < 72:
        return 'Class E'
    elif x < 448:
        return 'Class D'
    elif x < 1428:
        return 'Class C'
    elif x < q99:
        return 'Class B'
    else:
        return 'Class A'
    
q99 = df['balance'].quantile(0.99)

# df['class'] = df['balance'].apply(lambda x: balanceator(x, q99))


def wrangle(path):
    df = pd.read_csv(path, sep=';')

    # quantile
    q99 = df['balance'].quantile(0.99)

    df['y'] = df['y'].apply(lambda x: x == 'yes')
    df['default'] = df['default'].apply(lambda x: x == 'yes')
    df['housing'] = df['housing'].apply(lambda x: x == 'yes')
    df['loan'] = df['loan'].apply(lambda x: x == 'yes')

    df['balance_class'] = df['balance'].apply(lambda x: balanceator(x, q99))

    df['previous_bool'] = df['previous'].apply(lambda x: x != 0)

    # drop columns
    to_drop = ['previous', 'day', 'poutcome', 'pdays']
    df.drop(columns=to_drop, inplace=True)

    return df

df_pos = wrangle(r"E:\Data Analysis Course\MACHINE LEARNING\MACHINE LEARNING REAL TIME PROJECT\Marketing Data Analysis\bank.csv")

X = df_pos.drop(columns = ["y", "balance", 'duration'])
y = df_pos['y']


oe = OrdinalEncoder()
X = oe.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)


dt = GridSearchCV(DecisionTreeClassifier(random_state=42), {}, n_jobs=-1, cv=10, refit="recall")
dt.fit(X_train, y_train)


ConfusionMatrixDisplay.from_estimator(dt,X_test,y_test)

ConfusionMatrixDisplay.from_estimator(dt,X_test,y_test)

pred = dt.predict(X_test)

print (classification_report(y_test, pred))


params_dt = {
    "max_depth": [5, 10, 15, 20, 25, 30, None], # Maximum depth of the decision tree
    "criterion": ["gini","entropy"], # The quality criterion to measure the information gain when splitting nodes
    "min_samples_split": [2,3], # Minimum number of samples required to split an internal node
    "min_samples_leaf": [1,2] # Minimum number of samples required to be at a leaf node
}



model_dt = GridSearchCV(
    DecisionTreeClassifier(random_state=42), # Define the Decision Tree model
    params_dt, # Pass in the hyperparameters to be tuned from the dictionary we defined earlier
    cv=10, # Set the number of folds for cross-validation
    verbose=2
)



model_dt.fit(X_train, y_train)


ConfusionMatrixDisplay.from_estimator(model_dt,X_test,y_test)


pred_dt = model_dt.predict(X_test)


print (classification_report(pred_dt, y_test))