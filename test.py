import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pykrx
from pykrx import stock as st
from datetime import datetime
import datetime as dt
import os
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import random
import json
from collections import OrderedDict
import pickle
from catboost import CatBoostClassifier
import zerorpc


class StockAI(object):
    def __init__(self):
        self.date_list = []
        self.df_list = []
        self.get_stock_list()
        self.get_kospi_stock_list()
        self.get_kosdaq_stock_list()
        #self.get_weekly_list()
        self.rf_list = []
        self.cb_list = []
        self.goal_list = []

    def generate_train_data(self, n, left_date, right_date, train_len, result_len, code_list):
        x_list = []
        y_closing_list = []
        y_highest_list = []

        while True:
            if len(x_list) == n:
                break

            duration = (datetime.strptime(right_date, '%Y%m%d') - datetime.strptime(left_date, '%Y%m%d')).days
            rand_offset = random.randrange(0, duration)

            start_date = datetime.strptime(left_date, '%Y%m%d') + dt.timedelta(days=rand_offset)
            end_date = (start_date + dt.timedelta(days=(train_len + result_len) * 2)).strftime("%Y%m%d")
            start_date = start_date.strftime("%Y%m%d")
            rand_code = code_list[random.randrange(0, len(code_list))]
            price_data = st.get_market_ohlcv_by_date(start_date, end_date, rand_code)

            if len(price_data['종가']) < train_len + result_len:
                continue

            closing_price_norm = price_data['종가'] / np.max(price_data['종가'][:train_len])
            highest_price_norm = price_data['고가'] / np.max(price_data['종가'][:train_len])

            train_data = closing_price_norm[:train_len]
            raw_closing_data = closing_price_norm[train_len:train_len + result_len]
            raw_highest_data = highest_price_norm[train_len:train_len + result_len]

            x_list.append(train_data)
            y_closing_list.append(raw_closing_data)
            y_highest_list.append(raw_highest_data)

        x_list = np.array(x_list)
        y_closing_list = np.array(y_closing_list)
        y_highest_list = np.array(y_highest_list)

        return x_list, y_closing_list, y_highest_list

    # Crawling이 되지 않아, 파일로 대체함
    def get_stock_list(self):
        self.stock_list = pd.read_csv("data/market.csv")


    def get_kospi_stock_list(self):
        self.stock_list = pd.read_csv("data/kospi_market.csv")


    def get_kosdaq_stock_list(self):
        self.stock_list = pd.read_csv("data/kosdaq_market.csv")


    def get_weekly_data(self):
        path = "data/"
        for f in os.listdir(path):
            final_path = path + f
            if not f.__contains__('.csv'):
                continue
            df = pd.read_csv(final_path)
            if len(df) > 20:
                for i in range(len(df)):
                    c_origin = df['차트(원본)'][i][1:-1].split()
                    c_origin = [float(j) for j in c_origin]
                    df['차트(원본)'][i] = c_origin
                    t_origin = df['종가(원본)'][i][1:-1].split()
                    t_origin = [float(j) for j in t_origin]
                    df['종가(원본)'][i] = t_origin
                    h_origin = df['고가(원본)'][i][1:-1].split()
                    h_origin = [float(j) for j in h_origin]
                    df['고가(원본)'][i] = h_origin
                    c = df['차트'][i][1:-1].split()
                    c = [float(j) for j in c]
                    df['차트'][i] = c
                    t = df['종가'][i][1:-1].split()
                    t = [float(j) for j in t]
                    df['종가'][i] = t
                    h = df['고가'][i][1:-1].split()
                    h = [float(j) for j in h]
                    df['고가'][i] = h
                self.date_list.append(f.split('.csv')[0])
                print (f)
                self.df_list.append(df)

    def train_with_random_forest(self):
        y_highest_list = np.load("data/y_highest_list.npy")
        x_list = np.load("data/x_list.npy")

        for i in range(0, 31):
            goal = 0.005 + (0.005 * i)
            y_list = []
            print(goal)
            for k, item in enumerate(y_highest_list):
                y_list.append(int(np.max(item) >= (1 + goal) * x_list[k][-1]))

            y_list = np.array(y_list)
            rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
            rf.fit(x_list, y_list)
            self.goal_list.append(goal)
            self.rf_list.append(rf)

        print ("Random Forest Train Completed")


    def train_with_catboost(self):
        y_highest_list = np.load("data/y_highest_list.npy")
        x_list = np.load("data/x_list.npy")

        for i in range(0, 31):
            goal = 0.005 + (0.005 * i)
            y_list = []
            print(goal)
            for k, item in enumerate(y_highest_list):
                y_list.append(int(np.max(item) >= (1 + goal) * x_list[k][-1]))

            y_list = np.array(y_list)
            cb = CatBoostClassifier(iterations=10000, learning_rate=1, depth=2)
            cb.fit(x_list, y_list)
            self.cb_list.append(cb)

        print("Catboost Train Completed")


stock = StockAI()
#s = zerorpc.Server(StockAI())
#s.bind("tcp://0.0.0.0:4243")
#s.run()
print (stock.get_weekly_data())


