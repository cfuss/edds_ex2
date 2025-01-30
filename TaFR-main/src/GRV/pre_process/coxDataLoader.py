# -*- coding: UTF-8 -*-
import logging
import argparse
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import torchtuples as tt

from pycox.models import CoxTime
import os

class coxDataLoader:
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='../GRV/data/',
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='kwai_1115',
                            help='Choose a dataset.')
        parser.add_argument('--prediction_dataset', type=str, default='',
                            help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default='\t',
                            help='sep of csv file.')
        parser.add_argument('--play_rate', type=int, default=1,
                            help='in_features')
        parser.add_argument('--pctr', type=int, default=0,
                            help='in_features')
        parser.add_argument('--start_time', type=int, default=14,
                            help='group hour')
        parser.add_argument('--mock_data', type=int, default=0,
                            help='generate mock data for development')
        return parser



    def __init__(self, args):
        dataFolder=args.path+args.dataset
        print(os.path.abspath(dataFolder))
        self.coxData=pd.read_csv("%s/cox.csv"%dataFolder)
        renameDict={'item_id':'photo_id'}
        for i in range(168):
            renameDict['ctr%d'%i]='click_rate%d'%i
            renameDict['exp%d'%i]='exposure%d'%i
        self.coxData.rename(renameDict,inplace=True,axis=1)


        # self.filtered_data()
        print(len(self.coxData),self.coxData.columns)
        self.labtrans=None
        self.play_rate=args.play_rate
        self.pctr=args.pctr
        self.start_time=args.start_time
        self.prediction_dataset=args.prediction_dataset
        self.mock_data = args.mock_data
        return

    def filtered_data(self):
        caredList=['photo_id']
        for i in range(self.start_time):
            caredList.append('click_rate%d'%(i))
            if self.play_rate:
                caredList.append('play_rate%d'%(i))
            if self.pctr:
                caredList.append('new_pctr%d'%(i))
        self.coxData=self.coxData[caredList]

    @staticmethod
    def get_mock_data(rows=15, seed=1234) -> pd.DataFrame:
        np.random.seed(seed)

        data = []

        for photo_id in range(1, rows + 1):
            ctr = np.round(np.random.uniform(0.04, 0.30, rows), 2)
            exp = np.random.randint(60, 211, rows)
            play_rate = np.round(np.random.uniform(0.30, 0.80, rows), 2)
            new_pctr = np.round(ctr * np.random.uniform(1.5, 2.5), 2)

            row = [photo_id, *ctr, *exp, *play_rate, *new_pctr]
            data.append(row)

        columns = ["photo_id"] + \
                  [f"click_rate{i}" for i in range(0, rows)] + \
                  [f"exp{i}" for i in range(0, rows)] + \
                  [f"play_rate{i}" for i in range(0, rows)] + \
                  [f"new_pctr{i}" for i in range(0, rows)]

        return pd.DataFrame(data, columns=columns)

    @staticmethod
    def get_kwai_data_to_cox():
        df = pd.read_csv(r"/Users/cfuss/dev/uni/EXDDS/edds_ex2/data/KuaiRec/KuaiRec 2.0/data/item_daily_features.csv")

        df['click_rate'] = df['like_cnt'] / df['show_cnt']
        df['exp'] = df['show_cnt']
        df['play_rate'] = df['play_cnt'] / df['show_cnt']
        df['new_pctr'] = df['click_rate'] * np.random.uniform(1.5, 2.5, size=len(df))
        df = df[['video_id', 'click_rate', 'exp', 'play_rate', 'new_pctr']]

        df["index"] = df.groupby("video_id").cumcount()
        df = df.pivot(index="video_id", columns="index").reset_index()
        df.columns = [f"{col[0]}{col[1]}" if col[1] else col[0] for col in df.columns]

        df = df.rename(columns={'video_id': 'photo_id',
                                'click_rate': 'click_rate0',
                                'play_rate': 'play_rate0',
                                'new_pctr': 'new_pctr0'})

        return df

    @staticmethod
    def get_died_info_data_from_kwai():
        df = pd.read_csv(r"C:\DS\repos\edds_ex2\data\KuaiRec\KuaiRec 2.0\data\item_daily_features.csv")
        # df = pd.read_csv(r"/Users/cfuss/dev/uni/EXDDS/edds_ex2/data/KuaiRec/KuaiRec 2.0/data/item_daily_features.csv")
        df["photo_id"] = df["video_id"]
        df["timelevel"] = df["date"]
        df["tag"] = df["video_tag_id"].astype(str)
        df["riskFlag"] = (df["report_cnt"] + df["reduce_similar_cnt"]).astype(float)
        df["risk"] = -df["riskFlag"]

        df["original_show_cnt"] = df.groupby("photo_id")["show_cnt"].transform("first")

        df["decrease_percent"] = (df["original_show_cnt"] - df["show_cnt"]) / df["original_show_cnt"]

        df["died"] = (
                (df["visible_status"] != 'public') |
                (df["decrease_percent"] >= 0.8)
        ).astype(int)

        df = df.groupby("photo_id").head(7)

        df = df[["photo_id", "timelevel", "tag", "riskFlag", "risk", "died"]]

        return df[df["died"] == 1].drop_duplicates(subset=["photo_id"])

    def load_data(self,args):
        # self.filtered_data()
        self.diedInfo = pd.read_csv(args.label_path + '\\kwai_1115__1__24__168__0.5__0.5__-3.csv.csv')
        # self.diedInfo = pd.read_csv(args.label_path + '/kwai_1115__1__24__168__0.5__0.5__-3.csv.csv')
        if self.mock_data == 0:
            df_train = self.get_kwai_data_to_cox()
            self.diedInfo = self.get_died_info_data_from_kwai()
        else:
            df_train = self.get_mock_data()
        df_train.fillna(-1,inplace=True)
        caredList=['died','timelevel','photo_id']
        for i in range(self.start_time):
            caredList.append('click_rate%d'%(i))
            if self.play_rate:
                caredList.append('play_rate%d'%(i))
            if self.pctr:
                caredList.append('new_pctr%d'%(i))
        #df_train=df_train.drop([])
        self.diedInfo['photo_id'] = self.diedInfo['photo_id'].astype(int)
        df_train=pd.merge(df_train,self.diedInfo[['photo_id','died','timelevel']],on='photo_id')
        logging.info('[coxDataLoader] before died filter %d items'%len(df_train))
        df_train=df_train[df_train['died']==1]
        logging.info(df_train[['died','timelevel']].describe())
        df_test = df_train.sample(frac=0.2)
        df_train = df_train.drop(df_test.index)
        df_val = df_train.sample(frac=0.2)
        df_train = df_train.drop(df_val.index)

        if self.prediction_dataset!='':
            x=args.label_path
            # prediction_label_path=x.replace(args.dataset,args.prediction_dataset)
            # diedInfo=pd.read_csv(prediction_label_path)
            coxData=pd.read_csv(args.path+args.prediction_dataset+'/cox.csv')
            coxData.fillna(-1,inplace=True)
            caredList=['died','timelevel','photo_id']
            for i in range(self.start_time):
                caredList.append('click_rate%d'%(i))
                if self.play_rate:
                    caredList.append('play_rate%d'%(i))
                if self.pctr:
                    caredList.append('new_pctr%d'%(i))
            coxData=coxData[caredList[2:]] 
            df_test=coxData
            print("********",len(df_test))
            df_test['died']=1
            df_test['timelevel']=168
            # df_test=pd.merge(coxData,diedInfo[['photo_id','died','timelevel']],on='photo_id')
            # print("length!!!",len(diedInfo),len(coxData),len(df_test))

        return df_train,df_val,df_test,caredList


    def preprocess(self,args):
        df_train,df_val,df_test,cared=self.load_data(args)
        ignore_length=3
        cols_standardize = cared[ignore_length:]
        # cols_leave = ['photo_id']
        #scaler.mean_,np.sqrt(scaler.var_)
        standardize = [([col], StandardScaler()) for col in cols_standardize]
        # leave = [(col, None) for col in cols_leave]
        x_mapper = DataFrameMapper(standardize)#+leave

        df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_train.fillna(-1, inplace=True)
        self.x_train = x_mapper.fit_transform(df_train).astype('float32')
        if len(df_val)>0:
            df_val.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_val.fillna(-1, inplace=True)
            x_val = x_mapper.transform(df_val).astype('float32')
        df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_test.fillna(-1, inplace=True)
        self.x_test = x_mapper.transform(df_test).astype('float32')
        self.df_test=df_test

        self.labtrans = CoxTime.label_transform()
        get_target = lambda df: (df['timelevel'].values, df['died'].values)
        self.y_train = self.labtrans.fit_transform(*get_target(df_train))
        y_val = self.labtrans.transform(*get_target(df_val))
        self.val = tt.tuplefy(x_val, y_val)
        self.durations_test, self.events_test = get_target(df_test)
        # self.y_test= self.corpus.labtrans(*get_target(df_test))

    
if __name__ == '__main__':
    print("[dataLoader]")
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser = coxDataLoader.parse_data_args(parser)
    args, extras = parser.parse_known_args()

    # args.path = '../../data/'
    corpus = coxDataLoader(args)
    # label=Label(corpus)
    # corpus.preprocess()

