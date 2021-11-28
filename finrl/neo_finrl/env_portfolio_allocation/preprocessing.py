from . import config
import os
import itertools
import pandas as pd

def preprocess_data(data_file="usdt_1h_ada_algo_btc_eth_tur.csv"):
    path = os.path.join(os.getcwd(), config.INPUT_DATA_DIR, data_file)
    data = pd.read_csv(path);
    list_ticker = data["tic"].unique().tolist()

    list_date = list(pd.date_range(data['date'].min(),data['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))

    processed_data = pd.DataFrame(combination,columns=["date","tic"]).merge(data,on=["date","tic"],how="left")
    processed_data = processed_data[processed_data['date'].isin(data['date'])]
    processed_data = processed_data.sort_values(['date','tic'])

    processed_data = processed_data.fillna(0)

    data = processed_data

    data = data.sort_values(["date", "tic"], ignore_index=True)
    data.index = data["date"].factorize()[0]
    return data 

# def divide_data(df):
	# time_frequency = 15, start = None, mid1 = 172197, mid2 = 216837, 
 #                 end = None,
        # n = price_ary.shape[0]
        # if self.mode == 'train':
        #     self.price_ary = price_ary[start:mid1]
        #     self.tech_ary = tech_ary[start:mid1]
        #     n = self.price_ary.shape[0]
        #     x = n//int(time_frequency)
        #     ind = [int(time_frequency)*i for i in range(x)]
        #     self.price_ary = self.price_ary[ind]
        #     self.tech_ary = self.tech_ary[ind]
        # elif self.mode == 'test':
        #     self.price_ary = price_ary[mid1:mid2]
        #     self.tech_ary = tech_ary[mid1:mid2]
        #     n = self.price_ary.shape[0]
        #     x = n//int(time_frequency)
        #     ind = [int(time_frequency)*i for i in range(x)]
        #     self.price_ary = self.price_ary[ind]
        #     self.tech_ary = self.tech_ary[ind]
        # elif self.mode == 'trade':
        #     self.price_ary = price_ary[mid2:end]
        #     self.tech_ary = tech_ary[mid2:end]
        #     n = self.price_ary.shape[0]
        #     x = n//int(time_frequency)
        #     ind = [int(time_frequency)*i for i in range(x)]
        #     self.price_ary = self.price_ary[ind]
        #     self.tech_ary = self.tech_ary[ind]
        # else:
        #     raise ValueError('Invalid Mode!')
        