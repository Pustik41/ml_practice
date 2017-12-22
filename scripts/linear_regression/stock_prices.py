import pandas as pd
import quandl
import math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle


style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
#df = pd.read_csv('C:/Users/oleksandr.pustovyi/Documents/Udacity/ml_practice/resources/test_stock_price.csv')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close','Adj. Volume']]


df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0      # high price percent
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0   # daily percent price change

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]               # new data frame with features

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01 * len(df)))          # if use real data need change

df['lable'] = df[forecast_col].shift(-forecast_out)     # predict price


X = np.array(df.drop(['lable'],1))      # frame with features
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]            #last 34
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['lable'])  # frame with lable

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2) # shuffle data

clf = LinearRegression()       #LinearRegresion algoritm option n_jobs = .., how many jobs
#clf = svm.SVR()                 #Support vector regression algoritm change kernel=.., default linear
clf.fit(X_train, y_train)

with open('linearregression.pickle', 'wb') as f:    # save clasefire after training
    pickle.dump(clf, f)


pickle_in = open('linearregression.pickle', 'wb')   # download classefire for using
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)

#print(accuracy)

# predict

forecast_set = clf.predict(X_lately)

print(X_lately)
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj. Close'].plot()
df['Forecats'].plot()
plt.legend(loc=4)
plt.xlable('Date')
plt.ylabel('Price')
plt.show()



