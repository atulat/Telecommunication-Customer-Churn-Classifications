# Importing necessaary libraries for importing data, perform EDA and Data manipulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NearMiss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# reading the csv file
df = pd.read_csv("/Users/atulat/Documents/Project_1/Churn.csv")
df.head()

# removing rhe Unnamed column
df.drop(columns="Unnamed: 0", inplace=True, axis='columns')

print("Shape of the Raw Data Frame : ", df.shape)

# Renaming Column names as it can cause issue while working/addressing a specific column
# Replace '.' with '_'
df.rename(columns={'area.code': 'A_code', 'account.length': 'Actv_Days', 'voice.plan': 'V_plan',
                   'voice.messages': 'V_msgs', 'intl.plan': 'intl_plan',
                   'intl.mins': 'intl_mins', 'intl.calls': 'intl_calls',
                   'intl.charge': 'intl_chrg', 'day.mins': 'day_mins',
                   'day.calls': 'day_calls', 'day.charge': 'day_chrg', 'eve.mins': 'eve_mins',
                   'eve.calls': 'eve_calls', 'eve.charge': 'eve_chrg', 'night.mins': 'night_mins',
                   'night.calls': 'night_calls',
                   'night.charge': 'night_chrg', 'customer.calls': 'cust_calls'}, inplace=True)

print("Info of the Dataframe : ",df.info())

df['V_plan'].replace(('yes', 'no'), (1, 0), inplace=True)
df['intl_plan'].replace(('yes', 'no'), (1, 0), inplace=True)
df['churn'].replace(('yes', 'no'), (1, 0), inplace=True)
for col in ['day_chrg', 'eve_mins']:
    df[col] = df[col].astype('float64')

# Checking for null values
print("Null Values : ",df.isna().sum())

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.distplot(df['day_chrg'])
plt.subplot(1, 2, 2)
sns.distplot(df['eve_mins'])

print("Day charge Skewness:", df["day_chrg"].skew(), " Kurtosis:", df["day_chrg"].kurtosis())
print("Evening minutes Skewness:", df["eve_mins"].skew(), " Kurtosis:", df["eve_mins"].kurtosis())

df['day_chrg'] = df['day_chrg'].fillna(df['day_chrg'].mean())
df['eve_mins'] = df['eve_mins'].fillna(df['eve_mins'].mean())

# Checking for duplicates
print("Duplicate Values : ", df[df.duplicated()])

# Churn Value for each Area Code
plt.figure(figsize=(17, 5), facecolor="yellow")
pd.crosstab(df["A_code"], df["churn"]).plot(kind="bar")

a = len(df[(df["V_plan"] == 0) & (df["churn"] == 1) & (df["intl_plan"] == 0)])
b = len(df[(df["V_plan"] == 0) & (df["churn"] == 1) & (df["intl_plan"] == 1)])
c = len(df[(df["intl_plan"] == 0) & (df["churn"] == 1) & (df["V_plan"] == 1)])
d = len(df[(df["intl_plan"] == 1) & (df["churn"] == 1) & (df["V_plan"] == 1)])
plt.figure(figsize=(12, 5), facecolor="lightgreen")
x = np.array(["452 \n Voice Plan = NO, \nInternational Plan = NO", "153\nVoice Plan = NO, \nInternational Plan = YES",
              "56 \nVoice Plan = YES, \nInternational Plan = NO", "46\n Voice Plan = YES, \nInternational Plan = YES"])
y = np.array([a, b, c, d])
plt.bar(x, y)
plt.show()

# density plot on all charges (International Charges, Day Charges, Evening Charges, Night Charges)
plt.figure(figsize=(17, 5), facecolor="pink")
sns.kdeplot(df['intl_chrg'], color='red')
sns.kdeplot(df['day_chrg'], color='blue')
sns.kdeplot(df['eve_chrg'], color='green')
sns.kdeplot(df['night_chrg'], color='black')

# Calculating all the average value for different charges respective of their states
# this is done to find out the top and bottom states, and their churn performance
avg_international_chrg = df.groupby(["state"])["intl_chrg"].mean()
sort_avg_international_chrg = avg_international_chrg.sort_values(ascending=True)
avg_day_chrg = df.groupby(["state"])["day_chrg"].mean()
sort_avg_day_chrg = avg_day_chrg.sort_values(ascending=True)
avg_eve_chrg = df.groupby(["state"])["eve_chrg"].mean()
sort_avg_eve_chrg = avg_eve_chrg.sort_values(ascending=True)
avg_night_chrg = df.groupby(["state"])["night_chrg"].mean()
sort_avg_night_chrg = avg_night_chrg.sort_values(ascending=True)

# Mean value of Day charges of their States
plt.figure(figsize=(17, 5), facecolor="lightblue")
my_range = range(0, len(sort_avg_day_chrg.index))
plt.stem(sort_avg_day_chrg)
(markers, stemlines, baseline) = plt.stem(sort_avg_day_chrg)
plt.setp(markers, marker='D', markersize=10, markeredgecolor="black", markeredgewidth=2, linewidth=0.5)
plt.xticks(my_range, sort_avg_day_chrg.index)
plt.show()

# Selecting the top 3 and the bottom 3 states and getting their Churn Performance for the Day Charges
plt.figure(figsize=(10, 4))
df[(df["state"] == 'MO') | (df["state"] == 'DC') | (df["state"] == 'IL')]['churn'].value_counts().plot(kind="barh")
plt.ylabel("Top 3 States and their Churn Performance")

plt.figure(figsize=(10, 4))
df[(df["state"] == 'KS') | (df["state"] == 'NJ') | (df["state"] == 'MD')]['churn'].value_counts().plot(kind="barh")
plt.ylabel("Last 3 States and their Churn Performance")

# Account Length and their BoxPlot
plt.figure(figsize=(25, 10), facecolor="lightblue")
sns.boxplot(x='state', y='Actv_Days', data=df)
plt.ylabel('Account Length')
plt.show()

# Swarm Plot on Churn Performance and their states according to their Account Days
plt.figure(figsize=(30, 15), facecolor="lightblue")
sns.swarmplot(x='state', y='Actv_Days', data=df, hue='churn')
plt.ylabel('Account Length')
plt.show()

# CHECKING FOR DATA IMBALANCE

counts = df.churn.value_counts()
counts.plot(kind='bar', rot=0)
plt.title("Churn Distribution")
plt.xticks(range(2), ['No', 'Yes'])
plt.xlabel("Churn")
plt.ylabel("Frequency")

# FEATURE ENGINEERING

l1 = df.columns

LE = LabelEncoder()
df["state"] = LE.fit_transform(df[["state"]])
df["A_code"] = LE.fit_transform(df[["A_code"]])

# We know that the given features shouldn't have negative value, so verifying the data
for i in l1:
    qmax = df[i].max()
    qmin = df[i].min()
    print(i)
    print("Upper Limit : ", qmax)
    print("Lower Limit : ", qmin)
    print("\n")

plt.figure(figsize=(20, 20))
plt.subplot(4, 4, 1)
sns.boxplot(y='Actv_Days', data=df)
plt.subplot(4, 4, 2)
sns.boxplot(y='V_msgs', data=df)
plt.subplot(4, 4, 3)
sns.boxplot(y='intl_mins', data=df)
plt.subplot(4, 4, 4)
sns.boxplot(y='intl_calls', data=df)
plt.subplot(4, 4, 5)
sns.boxplot(y='intl_chrg', data=df)
plt.subplot(4, 4, 6)
sns.boxplot(y='day_mins', data=df)
plt.subplot(4, 4, 7)
sns.boxplot(y='day_calls', data=df)
plt.subplot(4, 4, 8)
sns.boxplot(y='day_chrg', data=df)

plt.figure(figsize=(20, 20))
plt.subplot(4, 4, 1)
sns.boxplot(y='eve_mins', data=df)
plt.subplot(4, 4, 2)
sns.boxplot(y='eve_calls', data=df)
plt.subplot(4, 4, 3)
sns.boxplot(y='eve_chrg', data=df)
plt.subplot(4, 4, 4)
sns.boxplot(y='night_mins', data=df)
plt.subplot(4, 4, 5)
sns.boxplot(y='night_calls', data=df)
plt.subplot(4, 4, 6)
sns.boxplot(y='night_chrg', data=df)
plt.subplot(4, 4, 7)
sns.boxplot(y='cust_calls', data=df)

l1 = ['Actv_Days', 'V_msgs', 'intl_mins', 'intl_calls', 'intl_chrg', 'day_mins', 'day_calls', 'day_chrg', 'eve_mins',
      'eve_calls', 'eve_chrg', 'night_mins', 'night_calls', 'night_chrg', 'cust_calls']
# Removing the Outliers from the dataframe which have 'NO' Churn Value
for i in l1:
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1
    for j in range(len(df)):
        if j not in df.index:
            continue
        if df['churn'][j] == 0:
            if (df[i][j] < (Q1 - (1.5 * IQR))) or (df[i][j] > (Q3 + (1.5 * IQR))):
                df = df.drop(j)

print("Dataframe after removing outliers : " ,df.shape)

# checking correlation
print("Coorelation : ",df.corr())

# droping the columns which have high correlated features
df = df.drop(['state', 'A_code', "intl_mins", "day_mins", "eve_mins", "night_mins"], axis=1)

mms = MinMaxScaler()
df[['V_msgs', 'intl_calls', 'intl_chrg', 'day_calls',
    'day_chrg', 'eve_calls', 'eve_chrg', 'night_calls',
    'night_chrg', 'cust_calls']] = mms.fit_transform(df[['V_msgs', 'intl_calls', 'intl_chrg', 'day_calls',
                                                         'day_chrg', 'eve_calls', 'eve_chrg', 'night_calls',
                                                         'night_chrg', 'cust_calls']])
print("Scaled Data",df.head())

X = df.drop(columns=['churn'])
Y = df['churn']

(X_train, X_test, y_train, y_test) = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=21)

us = NearMiss()
X_US, Y_US = us.fit_resample(X_train, y_train)

df_temp = X_US.join(Y_US)
counts = df_temp.churn.value_counts()
counts.plot(kind='bar', rot=0)
plt.title("Churn Distribution")
plt.xticks(range(2), ['No', 'Yes'])
plt.xlabel("Churn")
plt.ylabel("Frequency")

OSam = RandomOverSampler(random_state=42)
X_OS, Y_OS = OSam.fit_resample(X_train, y_train)

df_temp = X_OS.join(Y_OS)
counts = df_temp.churn.value_counts()
counts.plot(kind='bar', rot=0)
plt.title("Churn Distribution")
plt.xticks(range(2), ['No', 'Yes'])
plt.xlabel("Churn")
plt.ylabel("Frequency")

smk = SMOTETomek(random_state=42)
X_SM, Y_SM = smk.fit_resample(X_train, y_train)

df_temp = X_SM.join(Y_SM)
counts = df_temp.churn.value_counts()
counts.plot(kind='bar', rot=0)
plt.title("Churn Distribution")
plt.xticks(range(2), ['No', 'Yes'])
plt.xlabel("Churn")
plt.ylabel("Frequency")

# Ploting the distribution of data points generated for each sampling techniques
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.title("Random Under Sampling")
df_temp1 = X_US.join(Y_US)
sns.scatterplot(x=X_US['Actv_Days'], y=X_US['intl_chrg'], hue=Y_US, data=df_temp1)
plt.subplot(2, 2, 2)
plt.title("Random Over Sampling")
df_temp2 = X_OS.join(Y_OS)
sns.scatterplot(x=X_OS['Actv_Days'], y=X_OS['intl_chrg'], hue=Y_OS, data=df_temp2)
plt.subplot(2, 2, 3)
plt.title("SMOTE")
df_temp3 = X_SM.join(Y_SM)
sns.scatterplot(x=X_SM['Actv_Days'], y=X_SM['intl_chrg'], hue=Y_SM, data=df_temp3)

# LOGISTIC REGRESSION
# without data balancing

print("LOGISTIC REGRESSION without Data Balancing")
LR = LogisticRegression()
LR.fit(X_train, y_train)
Y_predtrain1 = LR.predict(X_train)
Y_predtest1 = LR.predict(X_test)

CM1 = confusion_matrix(y_train, Y_predtrain1)
print("confusion_matrix:\n", CM1)

AC_train1 = accuracy_score(y_train, Y_predtrain1)
print("accuracy of train data:\n", round(AC_train1 * 100, 2))

AC_test1 = accuracy_score(y_test, Y_predtest1)
print("accuracy of test data:\n", round(AC_test1 * 100, 2))

CR1 = classification_report(y_train, Y_predtrain1)
print("Report of train: \n", CR1)

CR_test1 = classification_report(y_test, Y_predtest1)
print("Report of test: \n", CR_test1)

# with balancing
print("LOGISTIC REGRESSION with Data Balancing")
LR.fit(X_SM, Y_SM)
Y_predtrain2 = LR.predict(X_SM)
Y_predtest2 = LR.predict(X_test)

CM2 = confusion_matrix(Y_SM, Y_predtrain2)
print("confusion_matrix:\n", CM2)

AC_train2 = accuracy_score(Y_SM, Y_predtrain2)
print("accuracy of train data:\n", round(AC_train2 * 100, 2))

AC_test2 = accuracy_score(y_test, Y_predtest2)
print("accuracy of test data:\n", round(AC_test2 * 100, 2))

CR2 = classification_report(Y_SM, Y_predtrain2)
print("Report of train: \n", CR2)

CR_test2 = classification_report(y_test, Y_predtest2)
print("Report of test: \n", CR_test2)

# NAIVE BAYES
# without data imbalancing

print("NAIVE BAYES without Data Balancing")
NB = MultinomialNB()
NB.fit(X_train, y_train)
Y_predtrain3 = NB.predict(X_train)
Y_predtest3 = NB.predict(X_test)

CM3 = confusion_matrix(y_train, Y_predtrain3)
print("confusion_matrix:\n", CM3)

AC_train3 = accuracy_score(y_train, Y_predtrain3)
print("accuracy of train data:\n", round(AC_train3 * 100, 2))

AC_test3 = accuracy_score(y_test, Y_predtest3)
print("accuracy of test data:\n", round(AC_test3 * 100, 2))

CR3 = classification_report(y_train, Y_predtrain3)
print("Report of train: \n", CR3)

CR_test3 = classification_report(y_test, Y_predtest3)
print("Report of test: \n", CR_test3)

# with balancing
print("NAIVE BAYES with Data Balancing")
NB.fit(X_SM, Y_SM)
Y_predtrain4 = NB.predict(X_SM)
Y_predtest4 = NB.predict(X_test)

CM4 = confusion_matrix(Y_SM, Y_predtrain4)
print("confusion_matrix:\n", CM4)

AC_train4 = accuracy_score(Y_SM, Y_predtrain4)
print("accuracy of train data:\n", round(AC_train4 * 100, 2))

AC_test4 = accuracy_score(y_test, Y_predtest4)
print("accuracy of test data:\n", round(AC_test4 * 100, 2))

CR4 = classification_report(Y_SM, Y_predtrain4)
print("Report of train: \n", CR4)

CR_test4 = classification_report(y_test, Y_predtest4)
print("Report of test: \n", CR_test4)

# DECISION TREE
# without data balancing
print("DECISION TREE without Data Balancing")
DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
Y_predtrain5 = DT.predict(X_train)
Y_predtest5 = DT.predict(X_test)

CM5 = confusion_matrix(y_train, Y_predtrain5)
print("confusion_matrix:\n", CM5)

AC_train5 = accuracy_score(y_train, Y_predtrain5)
print("accuracy of train data:\n", round(AC_train5 * 100, 2))

AC_test5 = accuracy_score(y_test, Y_predtest5)
print("accuracy of test data:\n", round(AC_test5 * 100, 2))

CR5 = classification_report(y_train, Y_predtrain5)
print("Report of train: \n", CR5)

CR_test5 = classification_report(y_test, Y_predtest5)
print("Report of test: \n", CR_test5)

# with balancing
print("DECISION TREE with Data Balancing")
DT.fit(X_SM, Y_SM)
Y_predtrain6 = DT.predict(X_SM)
Y_predtest6 = DT.predict(X_test)

CM6 = confusion_matrix(Y_SM, Y_predtrain6)
print("confusion_matrix:\n", CM6)

AC_train6 = accuracy_score(Y_SM, Y_predtrain6)
print("accuracy of train data:\n", round(AC_train6 * 100, 2))

AC_test6 = accuracy_score(y_test, Y_predtest6)
print("accuracy of test data:\n", round(AC_test6 * 100, 2))

CR6 = classification_report(Y_SM, Y_predtrain6)
print("Report of train: \n", CR6)

CR_test6 = classification_report(y_test, Y_predtest6)
print("Report of test: \n", CR_test6)

# GRADIENT BOOSTING
# without data balancing
print("GRADIENT BOOSTING without Data Balancing")
GBR = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100)

GBR.fit(X_train, y_train)
Y_predtrain7 = GBR.predict(X_train)
Y_predtest7 = GBR.predict(X_test)

CM7 = confusion_matrix(y_train, Y_predtrain7)
print("confusion_matrix:\n", CM7)

AC_train7 = accuracy_score(y_train, Y_predtrain7)
print("accuracy of train data:\n", round(AC_train7 * 100, 2))

AC_test7 = accuracy_score(y_test, Y_predtest7)
print("accuracy of test data:\n", round(AC_test7 * 100, 2))

CR7 = classification_report(y_train, Y_predtrain7)
print("Report of train: \n", CR7)

CR_test7 = classification_report(y_test, Y_predtest7)
print("Report of test: \n", CR_test7)

# with data balancing
print("GRADIENT BOOSTING with Data Balancing")
GBR.fit(X_SM, Y_SM)
Y_predtrain8 = GBR.predict(X_SM)
Y_predtest8 = GBR.predict(X_test)

CM8 = confusion_matrix(Y_SM, Y_predtrain8)
print("confusion_matrix:\n", CM8)

AC_train8 = accuracy_score(Y_SM, Y_predtrain8)
print("accuracy of train data:\n", round(AC_train8 * 100, 2))

AC_test8 = accuracy_score(y_test, Y_predtest8)
print("accuracy of test data:\n", round(AC_test8 * 100, 2))

CR8 = classification_report(Y_SM, Y_predtrain8)
print("Report of train: \n", CR8)

CR_test8 = classification_report(y_test, Y_predtest8)
print("Report of test: \n", CR_test8)

# RANDOM FOREST
# without data balancing
print("RANDOM FOREST without Data Balancing")
RF = RandomForestClassifier(max_depth=6, n_estimators=400, max_features=10)
RF.fit(X_train, y_train)
Y_predtrain9 = RF.predict(X_train)
Y_predtest9 = RF.predict(X_test)

CM9 = confusion_matrix(y_train, Y_predtrain9)
print("confusion_matrix:\n", CM9)

AC_train9 = accuracy_score(y_train, Y_predtrain9)
print("accuracy of train data:\n", round(AC_train9 * 100, 2))

AC_test9 = accuracy_score(y_test, Y_predtest9)
print("accuracy of test data:\n", round(AC_test9 * 100, 2))

CR9 = classification_report(y_train, Y_predtrain9)
print("Report of train: \n", CR9)

CR_test9 = classification_report(y_test, Y_predtest9)
print("Report of test: \n", CR_test9)

# with balancing
print("RANDOM FOREST with Data Balancing")
RF.fit(X_SM, Y_SM)
Y_predtrain10 = RF.predict(X_SM)
Y_predtest10 = RF.predict(X_test)

CM10 = confusion_matrix(Y_SM, Y_predtrain10)
print("confusion_matrix:\n", CM10)

AC_train10 = accuracy_score(Y_SM, Y_predtrain10)
print("accuracy of train data:\n", round(AC_train10 * 100, 2))

AC_test10 = accuracy_score(y_test, Y_predtest10)
print("accuracy of test data:\n", round(AC_test10 * 100, 2))

CR10 = classification_report(Y_SM, Y_predtrain10)
print("Report of train: \n", CR10)

CR_test10 = classification_report(y_test, Y_predtest10)
print("Report of test: \n", CR_test10)

# SUPPORT VECTOR MACHINE


clf1 = SVC(kernel='linear', gamma=0.001)
clf1.fit(X_train, y_train)
y_pred = clf1.predict(X_test)

acc = accuracy_score(y_test, y_pred) * 100

print("Accuracy with linear kernel :", round(acc, 2))

# without data balancing

# Final Model
print("SUPPORT VECTOR MACHINE without Data Balancing")
clf_final = SVC(kernel='linear', gamma=50, C=10)
clf_final.fit(X_train, y_train)
Y_predtrain11 = clf_final.predict(X_train)
Y_predtest11 = clf_final.predict(X_test)

CM11 = confusion_matrix(y_train, Y_predtrain11)
print("confusion_matrix:\n", CM11)

AC_train11 = accuracy_score(y_train, Y_predtrain11)
print("accuracy of train data:\n", round(AC_train11 * 100, 2))

AC_test11 = accuracy_score(y_test, Y_predtest11)
print("accuracy of test data:\n", round(AC_test11 * 100, 2))

CR11 = classification_report(y_train, Y_predtrain11)
print("Report of train: \n", CR11)

CR_test11 = classification_report(y_test, Y_predtest11)
print("Report of test: \n", CR_test11)

# with data balancing
print("SUPPORT VECTOR MACHINE with Data Balancing")
clf_final.fit(X_SM, Y_SM)
Y_predtrain12 = clf_final.predict(X_SM)
Y_predtest12 = clf_final.predict(X_test)

CM12 = confusion_matrix(Y_SM, Y_predtrain12)
print("confusion_matrix:\n", CM12)

AC_train12 = accuracy_score(Y_SM, Y_predtrain12)
print("accuracy of train data:\n", round(AC_train12 * 120, 2))

AC_test12 = accuracy_score(y_test, Y_predtest12)
print("accuracy of test data:\n", round(AC_test12 * 120, 2))

CR12 = classification_report(Y_SM, Y_predtrain12)
print("Report of train: \n", CR12)

CR_test12 = classification_report(y_test, Y_predtest12)
print("Report of test: \n", CR_test12)

# K NEAREST NEIGHBOUR


# Hyperparameter Tuning
print("KNN Hyperparamter tuning")
# choose k between 1 to 41
k_range = [2 * i + 1 for i in range(0, 20)]
k_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, Y, cv=10)
    k_scores.append(scores.mean())
# plot to see clearly
plt.figure(figsize=(10, 7), facecolor='lightblue')
plt.bar(k_range, k_scores)
plt.plot(k_range, k_scores, color="red")
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.xticks(k_range)
plt.ylim(0.6, 1)
plt.show()

# Without data balancing
print("KNN without Data Balancing")
model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train, y_train)
Y_predtrain13 = model.predict(X_train)
Y_predtest13 = model.predict(X_test)

CM13 = confusion_matrix(y_train, Y_predtrain13)
print("confusion_matrix:\n", CM13)

AC_train13 = accuracy_score(y_train, Y_predtrain13)
print("accuracy of train data:\n", round(AC_train13 * 100, 2))

AC_test13 = accuracy_score(y_test, Y_predtest13)
print("accuracy of test data:\n", round(AC_test13 * 100, 2))

CR13 = classification_report(y_train, Y_predtrain13)
print("Report of train: \n", CR13)

CR_test13 = classification_report(y_test, Y_predtest13)
print("Report of test: \n", CR_test13)

# With data balancing
print("KNN with Data Balancing")
model.fit(X_SM, Y_SM)
Y_predtrain14 = model.predict(X_SM)
Y_predtest14 = model.predict(X_test)

CM14 = confusion_matrix(Y_SM, Y_predtrain14)
print("confusion_matrix:\n", CM14)

AC_train14 = accuracy_score(Y_SM, Y_predtrain14)
print("accuracy of train data:\n", round(AC_train14 * 140, 2))

AC_test14 = accuracy_score(y_test, Y_predtest14)
print("accuracy of test data:\n", round(AC_test14 * 140, 2))

CR14 = classification_report(Y_SM, Y_predtrain14)
print("Report of train: \n", CR14)

CR_test14 = classification_report(y_test, Y_predtest14)
print("Report of test: \n", CR_test14)

# Comparison of Accuracies of Models With and Without Sampling
print("Comparison of Accuracies of Models With and Without Sampling")
Acc = pd.DataFrame(data={
    'Algorithm': ['Logistic Regression', 'Naive Bayes', 'Decision Tree', 'Gradient Boosting', 'Random Forest', 'SVM',
                  'KNN'],
    'Without Balancing': [AC_test1, AC_test3, AC_test5, AC_test7, AC_test9, AC_test11, AC_test13],
    'With Balancing': [AC_test2, AC_test4, AC_test6, AC_test8, AC_test10, AC_test12, AC_test14]})

print(Acc)

# Comparison of Train and Test Accuracies of Models
print("Comparison of Train and Test Accuracies of Models")
New_Acc = pd.DataFrame(data={
    'Algorithm': ['Logistic Regression', 'Naive Bayes', 'Decision Tree', 'Gradient Boosting', 'Random Forest', 'SVM',
                  'KNN'],
    'Train Accuracy': [AC_train1, AC_train3, AC_train5, AC_train7, AC_train9, AC_train11, AC_train13],
    'Test Accuracy': [AC_test1, AC_test3, AC_test5, AC_test7, AC_test9, AC_test11, AC_test13]})

print(New_Acc)
# Gradient Boosting Algorithm is choosen for our Model and Streamlit Framework is used to deploy the model

print("Gradient Boosting Algorithm is chosen for our Model")
