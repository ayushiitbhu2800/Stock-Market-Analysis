import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from sklearn.metrics import f1_score
import pandas as pd
import logging
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification
from collections import Counter
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

from google.colab import drive
drive.mount('/content/drive')
input_file = '/content/drive/MyDrive/Tick.by.tick.data.csv'
data=[]
count=0
with open(input_file, newline='') as file:
    reader = csv.reader(file)
    for row in reader:
      count+=1
      data.append(row)

def get_formatted_data(data):
  d=[]
  for i in data:
    if i==None:
      continue
    x=i[0]
    d.append(x.split("|"))
  return d
def segregate_T_data(data):
  t=[]
  z=[]
  for i in data:
    if i==None:
      continue
    if i[0]=='T':
      t.append(np.array(i))
    elif len(i)==7:
      z.append(np.array(i))
  return z,t

data=get_formatted_data(data)
data,trade_data=segregate_T_data(data)

header=data[0]
header[0]="type"
data=pd.DataFrame(data[4:],columns=header,dtype="object")

def refractor_data(data):
    print("Before dropping columns and rows:")
    print(data.head())

    data = data.drop(columns=["order.id", "token"])
    type_mapping = {'N': 0, 'M': 1, 'X': 2}
    data['type'] = data['type'].map(type_mapping)
    data.drop(data[data['price'] == float(0)].index, inplace=True)

    print("After dropping columns and rows:")
    print(data.head())
    data["timestamp"]=data["timestamp"].astype("int64")
    data["timestamp"]=pd.to_datetime(data["timestamp"])
    return data

data=refractor_data(data)

data["price"]=data["price"].astype("float")
data["quantity"]=data["quantity"].astype("float")

trade_data=pd.DataFrame(trade_data)
trade_data[0]=3

trade_data=trade_data.drop(columns=[2,3,4])

trade_data.columns=["type","timestamp","price","quantity"]
trade_data["timestamp"]=trade_data["timestamp"].astype("int64")
trade_data["timestamp"]=pd.to_datetime(trade_data["timestamp"])
data=data.append(trade_data)
data=data.sort_values(by="timestamp")

data["order.type"]=data["order.type"].astype("str")
data["price"]=data["price"].astype("float64")
data["quantity"]=data["quantity"].astype("float64")

data.reset_index(drop=True)

class TechnicalIndicators:
    def __init__(self, data):
        self.data = data

    def calculate_sma(self, column, n):
        return self.data[column].rolling(window=n).mean()

    def calculate_macd(self, column):
        ema12 = self.data[column].ewm(span=12, adjust=False).mean()
        ema26 = self.data[column].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9, adjust=False).mean()
        return macd, signal_line

    def calculate_stochastic(self, column, n):
        high_n = self.data[column].rolling(window=n).max()
        low_n = self.data[column].rolling(window=n).min()
        rsv = 100 * (self.data[column] - low_n) / (high_n - low_n)
        k = pd.Series(50, index=self.data.index)
        d = pd.Series(50, index=self.data.index)
        for i in range(n, len(self.data)):
            k.iloc[i] = (2 / 3) * rsv.iloc[i] + (1 / 3) * k.iloc[i-1]
            d.iloc[i] = (2 / 3) * k.iloc[i] + (1 / 3) * d.iloc[i-1]
        j = 3 * d - 2 * k
        return k, d, j

    def calculate_rsi(self, column, n):
        delta = self.data[column].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=n).mean()
        avg_loss = loss.rolling(window=n).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
        return rsi

    def calculate_bias(self, column, n):
        ma_n = self.data[column].rolling(window=n).mean()
        bias = (self.data[column] - ma_n) / ma_n * 100
        return bias

    def calculate_obv(self, quantity_column, order_type_column):
        obv = self.data[quantity_column].where(self.data[order_type_column] == 'B', -self.data[quantity_column])
        obv = obv.cumsum()
        return obv

    def calculate_cr(self, column, n):
        delta = self.data[column].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        rolling_gain = gain.rolling(window=n).sum()
        rolling_loss = loss.rolling(window=n).sum()
        bull_strength = rolling_gain.rolling(window=n).sum()
        bear_strength = rolling_loss.rolling(window=n).sum()
        cr = bull_strength / bear_strength
        return cr

    def calculate_cr_ma(self, cr_column, n):
        return self.data[cr_column].rolling(window=n).mean()

    def calculate_vr(self, quantity_column, n):
        avs = self.data[quantity_column].iloc[:n].mean()
        bvs = self.data[quantity_column].iloc[n:2*n].mean()
        cvs = self.data[quantity_column].iloc[2*n:].mean()
        vr = (avs + cvs) / (2 * bvs + cvs)
        return vr

    def calculate_directionals(self, column, n):
        high_n = self.data[column].rolling(window=n).max()
        low_n = self.data[column].rolling(window=n).min()
        high_n1 = self.data[column].shift(1).rolling(window=n).max()
        low_n1 = self.data[column].shift(1).rolling(window=n).min()

        dm_plus = high_n - high_n1
        dm_plus[dm_plus <= 0] = 0

        dm_minus = low_n1 - low_n
        dm_minus[dm_minus <= 0] = 0

        true_range = self.data[column] - self.data[column]

        true_range[true_range < abs(self.data[column] - self.data[column].shift(1))] = abs(self.data[column] - self.data[column].shift(1))

        di_plus = dm_plus.rolling(window=n).sum() / true_range.rolling(window=n).sum() * 100
        di_minus = dm_minus.rolling(window=n).sum() / true_range.rolling(window=n).sum() * 100

        di_x = ((di_minus - di_plus).abs() / (di_minus + di_plus)).fillna(0) * 100

        return di_plus, di_minus, di_x

def get_indicators(data,n):
  tech_indicators = TechnicalIndicators(data)

  # Calculate technical indicators for each message type
  n_sma = n
  data['SMA5'] = tech_indicators.calculate_sma('price', n_sma)
  data["SMA20"] = tech_indicators.calculate_sma('price', n+15)

  macd, signal_line = tech_indicators.calculate_macd('price')
  data['MACD'] = macd
  data['Signal_Line'] = signal_line

  n_stochastic = 14
  k, d, j = tech_indicators.calculate_stochastic('price', n_stochastic)
  data['k'] = k
  data['d'] = d
  data['j'] = j

  n_rsi = n+1
  data['RSI'] = tech_indicators.calculate_rsi('price', n_rsi)

  n_bias = n
  data['BIAS'] = tech_indicators.calculate_bias('price', n_bias)

  data['OBV'] = tech_indicators.calculate_obv('quantity', 'order.type')

  n_cr = 26
  data['cr'] = tech_indicators.calculate_cr('price', n_cr)
  n_cr_ma = n
  data['CR-MA'] = tech_indicators.calculate_cr_ma('cr', n_cr_ma)

  n_vr = n
  data['VR'] = tech_indicators.calculate_vr('quantity', n_vr)
  n_di = n
  data['+DI'], data["-DI"], data["+DIX"] = tech_indicators.calculate_directionals('price', n_di)

def draw(s,t,u=-1,p=-1,xul=-1,yul=-1,xll=-1,yll=-1):
  if u==-1:
    plt.plot(data[s], data[t], linestyle='-', color='r')
  else:
    plt.plot(data[s].iloc[u:p], data[t].iloc[u:p], linestyle='-', color='r')
  plt.plot(data[s].iloc[u:p],data[t].iloc[u:p])
  plt.xlabel(s)
  plt.ylabel(t)
  plt.title(s+"vs." +t)
  if xul!=-1:
    plt.xlim(xll,xul)
  if yul!=-1:
    plt.ylim(yll,yul)
  plt.grid(True)
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.legend()
  plt.show()

def pnl(data,z):
  sum=0
  quantity=0
  y_pred=[]
  buy=0
  sell=0
  for i in range(len(data)):
    if data["type"].iloc[i]!=3:
      y_pred.append(3)
      continue
    if (data["SMA5"].iloc[i]>data["SMA20"].iloc[i]) and ((data["MACD"].iloc[i]-data["Signal_Line"].iloc[i])>0.1) and (data["MACD"].iloc[i]>0) and data["price"].iloc[i]>data["price"].iloc[i-1]:
      if data["quantity"].iloc[i]<z:
        sum-=data["price"].iloc[i]*data["quantity"].iloc[i]
        quantity+=data["quantity"].iloc[i]
        y_pred.append(0)
        buy+=1
      else:
        y_pred.append(2)
    elif data["SMA5"].iloc[i]<data["SMA20"].iloc[i]:
      if quantity>=data["quantity"].iloc[i]:
        sum+=data["price"].iloc[i]*data["quantity"].iloc[i]
        quantity-=data["quantity"].iloc[i]
        y_pred.append(1)
        sell+=1
      else:
        y_pred.append(2)
    else:
      y_pred.append(2)
  return sum,buy,sell,y_pred

def get_train_test_data(data):
  sum,buy,sell,y_pred=pnl(data,101)
  y_p=np.array(y_pred)
  x=data[data["type"]==3]
  y=y_p[y_p<=2]
  x=x.drop(columns=["timestamp","type","order.type","k","d","j","OBV","cr"])
  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42,shuffle=False)
  y_test=pd.DataFrame(y_test)
  y_train=pd.DataFrame(y_train)
  y_train=y_train.fillna(method="bfill")
  y_train=y_train.fillna(method="ffill")
  y_test=y_test.fillna(method="bfill")
  y_test=y_test.fillna(method="ffill")
  X_test=X_test.fillna(method="bfill")
  X_test=X_test.fillna(method="ffill")
  X_train=X_train.fillna(method="bfill")
  X_train=X_train.fillna(method="ffill")
  return X_train, X_test, y_train, y_test

class MLModel:
    def train_models(self, X_train, y_train):
      accuracy_logistic = self.train_logistic_regression(X_train,y_train)
      accuracy_lgbm = self.train_lgbm(X_train,y_train)
      accuracy_forest = self.train_random_forest(X_train,y_train)
      return {
          "logistic":accuracy_logistic,
          "lgbm":accuracy_lgbm,
          "forest":accuracy_forest
          }

    def train_logistic_regression(self, X_train, y_train,X_test,y_test):
      model = LogisticRegression()
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)
      accuracy = accuracy_score(y_test, y_pred)
      print(f"Accuracy: {accuracy}")
      print("Logistic Regression Report")
      print(classification_report(y_test, y_pred))
      return accuracy

    def train_random_forest(self, X_train, y_train,X_test,y_test):
        model = RandomForestClassifier()
        param_grid = {
          'n_estimators': [100, 200, 300],
          'max_depth': [None, 10, 20],
          'min_samples_split': [2, 5, 10],
          'min_samples_leaf': [1, 2, 4]
        }

        # param_grid = {
        #   'n_estimators': [100],
        #   'max_depth': [20],
        #   'min_samples_split': [10],
        #   'min_samples_leaf': [110]
        # }

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

        grid_search.fit(X_train, y_train)
        print("Best Hyperparameters:")
        print(grid_search.best_params_)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')
        print(classification_report(y_test, y_pred))
        return accuracy


    def train_lgbm(self, X_train, y_train,X_test,y_test):
      train_data = lgb.Dataset(X_train, label=y_train)
      param_grid = {
          'num_leaves': [15, 31, 63],
          'max_depth': [50],
          'learning_rate': [0.1, 0.01]
      }
      model = lgb.LGBMClassifier()
      grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
      grid_search.fit(X_train, y_train)
      print("Best Hyperparameters:")
      print(grid_search.best_params_)
      best_model = grid_search.best_estimator_
      y_pred = best_model.predict(X_test)
      accuracy = accuracy_score(y_test, y_pred)
      print(f'Accuracy: {accuracy:.2f}')
      return accuracy

    def train_undersample(self, X_train, y_train,X_test,y_test):
      undersampler = RandomUnderSampler(random_state=42)
      X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
      X_test_resampled, y_test_resampled = undersampler.fit_resample(X_test, y_test)
      accuracy_logistic = self.train_logistic_regression(X_train_resampled,y_train_resampled,X_test_resampled,y_test_resampled)
      accuracy_lgbm = self.train_lgbm(X_train_resampled,y_train_resampled,X_test_resampled,y_test_resampled)
      accuracy_forest=self.train_random_forest(X_train_resampled,y_train_resampled,X_test_resampled,y_test_resampled)
      return {
          "undersampled logistic":accuracy_logistic,
          "undersampled lgbm":accuracy_lgbm,
          "undersampled forest":accuracy_forest
          }

    def train_oversample(self, X_train, y_train,X_test,y_test):
      oversampler = RandomOverSampler(random_state=42)
      X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)
      X_test_resampled, y_test_resampled = oversampler.fit_resample(X_test, y_test)
      accuracy_logistic = self.train_logistic_regression(X_train_resampled,y_train_resampled,X_test_resampled,y_test_resampled)
      accuracy_lgbm = self.train_lgbm(X_train_resampled,y_train_resampled,X_test_resampled,y_test_resampled)
      accuracy_forest=self.train_random_forest(X_train_resampled,y_train_resampled,X_test_resampled,y_test_resampled)
      return {
          "oversampled logistic":accuracy_logistic,
          "oversampled lgbm":accuracy_lgbm,
          "oversampled forest":accuracy_forest
          }

    def kfold_xgb(self,X_train,y_train):
      oversampler = RandomOverSampler(random_state=42)
      classifier = xgb
      num_folds = 5
      skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
      accuracies = []
      profits = []
      for train_index, test_index in skf.split(X_train.drop(columns=["RSI","BIAS"]), y_train):
          X_train_fold, X_val_fold = X_train.drop(columns=["RSI","BIAS"]).iloc[train_index], X_train.drop(columns=["RSI","BIAS"]).iloc[test_index]
          y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]
          dtrain = lgb.Dataset(X_train_fold, label=y_train_fold)
          dtest = lgb.Dataset(X_val_fold, label=y_val_fold, reference=dtrain)
          params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'max_depth': 3,
            'learning_rate': 0.1,
            'num_leaves': 31
          }
          num_round = 100
          bst = lgb.train(params, dtrain, num_round, valid_sets=[dtest], early_stopping_rounds=10)
          y_pred = bst.predict(X_val_fold, num_iteration=bst.best_iteration)
          y_pred_max = [list(x).index(max(x)) for x in y_pred]
          accuracy = accuracy_score(y_val_fold, y_pred_max)
          sum=0
          quantity=0
          for j in range(len(y_pred_max)):
            if y_pred_max[j]==0:
              sum-=X_val_fold["price"].iloc[j]*X_val_fold["quantity"].iloc[j]
              quantity+=X_val_fold["quantity"].iloc[j]
            else:
              if quantity>=X_val_fold["quantity"].iloc[j]:
                sum+=X_val_fold["price"].iloc[j]*X_val_fold["quantity"].iloc[j]
                quantity-=X_val_fold["quantity"].iloc[j]
          profits.append(sum)
      print("Profit for k fold XGBoost")
      print(np.array(profits)/len(profits))
      return accuracy

    def kfold_logistic(self,X,y):
      num_folds = 5
      skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
      accuracies = []
      profit_results = []
      for train_index, test_index in skf.split(X, y):
          X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[test_index]
          y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[test_index]
          model = LogisticRegression()
          model.fit(X_train_fold.drop(columns=["RSI", "BIAS"]), y_train_fold)
          y_pred_val = model.predict(X_val_fold.drop(columns=["RSI", "BIAS"]))
          accuracy = accuracy_score(y_val_fold, y_pred_val)
          accuracies.append(accuracy)
          sum = 0
          quantity = 0
          for j in range(len(y_pred_val)):
              if y_pred_val[j] == 0:
                  sum -= X_val_fold["price"].iloc[j] * X_val_fold["quantity"].iloc[j]
                  quantity += X_val_fold["quantity"].iloc[j]
              else:
                  if quantity >= X_val_fold["quantity"].iloc[j]:
                      sum += X_val_fold["price"].iloc[j] * X_val_fold["quantity"].iloc[j]
                      quantity -= X_val_fold["quantity"].iloc[j]
              profit_results.append(sum)
      average_profits = np.mean(profit_results)
      print("Profit for k fold Logistic Regression")
      print(average_profits)
      return np.mean(accuracies)

    def kfold_random_forest(self,X,y):
      num_folds = 5
      skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
      accuracies = []
      profit_results = []
      for train_index, test_index in skf.split(X, y):
          X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[test_index]
          y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[test_index]
          model = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=10, min_samples_leaf=10)
          model.fit(X_train_fold.drop(columns=["RSI", "BIAS"]), y_train_fold)
          y_pred_val = model.predict(X_val_fold.drop(columns=["RSI", "BIAS"]))
          accuracy = accuracy_score(y_val_fold, y_pred_val)
          accuracies.append(accuracy)
          sum = 0
          quantity = 0
          for j in range(len(y_pred_val)):
              if y_pred_val[j] == 0:
                  sum -= X_val_fold["price"].iloc[j] * X_val_fold["quantity"].iloc[j]
                  quantity += X_val_fold["quantity"].iloc[j]
              else:
                  if quantity >= X_val_fold["quantity"].iloc[j]:
                      sum += X_val_fold["price"].iloc[j] * X_val_fold["quantity"].iloc[j]
                      quantity -= X_val_fold["quantity"].iloc[j]
              profit_results.append(sum)
      average_profits = np.mean(profit_results)
      print("Average Profits for k fold Random Forest")
      print(average_profits)
      return np.mean(accuracies)

    def kfold_lgbm(self,X,y):
      num_folds = 5
      skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
      accuracies = []
      profit_results = []
      for train_index, test_index in skf.split(X, y):
          X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[test_index]
          y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[test_index]
          dtrain = lgb.Dataset(X_train_fold.drop(columns=["RSI", "BIAS"]), label=y_train_fold)
          dtest = lgb.Dataset(X_val_fold.drop(columns=["RSI", "BIAS"]), label=y_val_fold, reference=dtrain)
          params = {
              'objective': 'multiclass',
              'num_class': 3,
              'metric': 'multi_logloss',
              'max_depth': 3,
              'learning_rate': 0.1,
              'num_leaves': 31
          }
          num_round = 100
          model = lgb.train(params, dtrain, num_round, valid_sets=[dtest], early_stopping_rounds=10)
          y_pred_probs = model.predict(X_val_fold.drop(columns=["RSI", "BIAS"]), num_iteration=model.best_iteration)
          y_pred_val = [list(x).index(max(x)) for x in y_pred_probs]
          accuracy = accuracy_score(y_val_fold, y_pred_val)
          accuracies.append(accuracy)
          sum = 0
          quantity = 0
          for j in range(len(y_pred_val)):
              if y_pred_val[j] == 0:
                  sum -= X_val_fold["price"].iloc[j] * X_val_fold["quantity"].iloc[j]
                  quantity += X_val_fold["quantity"].iloc[j]
              else:
                  if quantity >= X_val_fold["quantity"].iloc[j]:
                      sum += X_val_fold["price"].iloc[j] * X_val_fold["quantity"].iloc[j]
                      quantity -= X_val_fold["quantity"].iloc[j]
              profit_results.append(sum)
      average_profits = np.mean(profit_results)
      print("Average Profits for k fold Random Forest")
      print(average_profits)
      return np.mean(accuracies)


    def k_fold(self,X_train,y_train):
      accuracy_xgb=self.kfold_xgb(X_train,y_train)
      accuracy_logistic=self.kfold_logistic(X_train,y_train)
      accuracy_xgb=self.kfold_random_forest(X_train,y_train)
      accuracy_xgb=self.kfold_lgbm(X_train,y_train)

model = MLModel()

window_5=data
get_indicators(window_5,5)
X_train,X_test,y_train,y_test=get_train_test_data(data)
undersample_accuracies_5 = model.train_undersample(X_train,y_train,X_test,y_test)
oversample_accuracies_5 = model.train_oversample(X_train,y_train,X_test,y_test)

window_10=data
get_indicators(window_10,10)
X_train,X_test,y_train,y_test=get_train_test_data(data)
undersample_accuracies_10 = model.train_undersample(X_train,y_train,X_test,y_test)
oversample_accuracies_10 = model.train_oversample(X_train,y_train,X_test,y_test)

window_15=data
get_indicators(window_15,15)
X_train,X_test,y_train,y_test=get_train_test_data(data)
undersample_accuracies_15 = model.train_undersample(X_train,y_train,X_test,y_test)
oversample_accuracies_15 = model.train_oversample(X_train,y_train,X_test,y_test)

window_20=data
get_indicators(window_20,20)
X_train,X_test,y_train,y_test=get_train_test_data(data)
undersample_accuracies_20 = model.train_undersample(X_train,y_train,X_test,y_test)
oversample_accuracies_20 = model.train_oversample(X_train,y_train,X_test,y_test)

window_25=data
get_indicators(window_25,25)
X_train,X_test,y_train,y_test=get_train_test_data(data)
undersample_accuracies_25 = model.train_undersample(X_train,y_train,X_test,y_test)
oversample_accuracies_25 = model.train_oversample(X_train,y_train,X_test,y_test)

print("for window size 5:")
print(undersample_accuracies_5)
print(oversample_accuracies_5)

print("for window size 10:")
print(undersample_accuracies_10)
print(oversample_accuracies_10)

print("for window size 15:")
print(undersample_accuracies_15)
print(oversample_accuracies_15)

print("for window size 20:")
print(undersample_accuracies_20)
print(oversample_accuracies_20)

print("for window size 25:")
print(undersample_accuracies_25)
print(oversample_accuracies_25)

def plot_accuracy(data):
    window_sizes = [5, 10, 15, 20, 25]

    # Define models and their corresponding colors
    models = {
        'logistic': 'blue',
        'lgbm': 'red',
        'forest': 'green'
    }

    for sampling_type in ['undersampled', 'oversampled']:
        plt.figure(figsize=(10, 5))

        for model, color in models.items():
            model_accuracies = [data[ws][sampling_type + ' ' + model] for ws in window_sizes]
            plt.plot(window_sizes, model_accuracies, label=model, color=color)

        plt.xlabel('Window Size')
        plt.ylabel('Accuracy')
        plt.title(f'{sampling_type.capitalize()}: Accuracy for Different Window Sizes')
        plt.xticks(window_sizes)
        plt.legend()
        plt.show()

accuracy_data = {
    5: {
        'undersampled logistic': undersample_accuracies_5['undersampled logistic'],
        'undersampled lgbm': undersample_accuracies_5['undersampled lgbm'],
        'undersampled forest': undersample_accuracies_5['undersampled forest'],
        'oversampled logistic': oversample_accuracies_5['oversampled logistic'],
        'oversampled lgbm': oversample_accuracies_5['oversampled lgbm'],
        'oversampled forest': oversample_accuracies_5['oversampled forest']
    },
    10: {
        'undersampled logistic': undersample_accuracies_10['undersampled logistic'],
        'undersampled lgbm': undersample_accuracies_10['undersampled lgbm'],
        'undersampled forest': undersample_accuracies_10['undersampled forest'],
        'oversampled logistic': oversample_accuracies_10['oversampled logistic'],
        'oversampled lgbm': oversample_accuracies_10['oversampled lgbm'],
        'oversampled forest': oversample_accuracies_10['oversampled forest']
    },
    15: {
        'undersampled logistic': undersample_accuracies_15['undersampled logistic'],
        'undersampled lgbm': undersample_accuracies_15['undersampled lgbm'],
        'undersampled forest': undersample_accuracies_15['undersampled forest'],
        'oversampled logistic': oversample_accuracies_15['oversampled logistic'],
        'oversampled lgbm': oversample_accuracies_15['oversampled lgbm'],
        'oversampled forest': oversample_accuracies_15['oversampled forest']
    },
    20: {
        'undersampled logistic': undersample_accuracies_20['undersampled logistic'],
        'undersampled lgbm': undersample_accuracies_20['undersampled lgbm'],
        'undersampled forest': undersample_accuracies_20['undersampled forest'],
        'oversampled logistic': oversample_accuracies_20['oversampled logistic'],
        'oversampled lgbm': oversample_accuracies_20['oversampled lgbm'],
        'oversampled forest': oversample_accuracies_20['oversampled forest']
    },
    25: {
        'undersampled logistic': undersample_accuracies_25['undersampled logistic'],
        'undersampled lgbm': undersample_accuracies_25['undersampled lgbm'],
        'undersampled forest': undersample_accuracies_25['undersampled forest'],
        'oversampled logistic': oversample_accuracies_25['oversampled logistic'],
        'oversampled lgbm': oversample_accuracies_25['oversampled lgbm'],
        'oversampled forest': oversample_accuracies_25['oversampled forest']
    }
}

plot_accuracy(accuracy_data)