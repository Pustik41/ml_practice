import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
import pandas as pd

'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''

data_path = "C:/Users/oleksandr.pustovyi/Documents/Udacity/ml_practice/resources/titanic.xls"
df = pd.read_excel(data_path)
original_df = pd.DataFrame.copy(df)

df.drop(['body', 'name'], 1, inplace = True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
#print(df.head())

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_val = {}
        def convert_to_int(val):
            return text_digit_val[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unq in unique_elements:
                if unq not in text_digit_val:
                    text_digit_val[unq] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(df)
df.drop(['boat'], 1, inplace=True)
#print(df.head())

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
clusters_centers = clf.cluster_centers_
n_clusters = len(np.unique(labels))

original_df['cluster_group'] = np.nan

for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]

survival_rates = {}
for i in range(n_clusters):
    temp_df = original_df[ (original_df['cluster_group'] == float(i)) ]
    survival_cluster = temp_df[ (temp_df['survived'] == 1) ]
    survival_rate = float(len(survival_cluster))/len(temp_df)
    survival_rates[i] = survival_rate

print(survival_rates)
print(original_df[ (original_df['cluster_group'] == 1)])
print(original_df[ (original_df['cluster_group'] == 0)].describe())

cluster_0 =original_df[ (original_df['cluster_group'] == 0)]
cluster_0_fc = cluster_0[ (cluster_0['pclass'] == 1) ]          # how many first class passengers survived in o cluster

print(cluster_0_fc.describe())











