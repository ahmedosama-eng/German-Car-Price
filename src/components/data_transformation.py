import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd 
from sklearnpython .compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
import os 

from src.utils import save_object,is_canot_float_convertible

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_trasfromation_config=DataTransformationConfig()

    def get_data_transformer_object(self,data):
        try:
            numiric_features=data.select_dtypes(exclude="object").columns
            categorical_features=data.select_dtypes(include="object").columns

            num_trasform=StandardScaler()
            cat_trasform=OrdinalEncoder()

            preprocessor=ColumnTransformer(
                [
                    ("OrdinalEncoder",cat_trasform,categorical_features),
                    ("StanderdScaler",num_trasform,numiric_features)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def cleaning_data(self,data):
        try:
            data.drop(['Unnamed:0','offer_description'],axis=1,inplace=True)
            condition=data[data['fuel_consumption_l_100km'].str.contains('-')]['fuel_consumption_l_100km'].index
            data.drop(index=condition,inplace=True)
            condition=data[data['fuel_consumption_g_km'].str.contains('-')]['fuel_consumption_g_km'].index
            data.drop(index=condition,inplace=True)
            condition=data[data['fuel_consumption_l_100km'].str.contains('Diesel')]['fuel_consumption_l_100km'].index
            data.drop(index=condition,inplace=True)
            condition=data[data['fuel_consumption_g_km'].str.contains('Petrol')]['fuel_consumption_g_km'].index
            data.drop(index=condition,inplace=True)
            data['fuel_consumption_l_100km']=(data['fuel_consumption_l_100km'].str.split(' ').str[0].str.replace(',','.')).astype(float)
            data['fuel_consumption_g_km']=(data['fuel_consumption_g_km'].str.split(' ').str[0].str.replace(',','.')).astype(float)
            data['power_kw']=(data['power_kw'].str.split(' ').str[0].str.replace(',','.')).astype(float)
            data['power_ps']=(data['power_ps'].str.split(' ').str[0].str.replace(',','.')).astype(float)
            condition = data[data['price_in_euro'].apply(is_canot_float_convertible)].index
            data.drop(index=condition,inplace=True)
            condition = data[data['year'].apply(is_canot_float_convertible)].index
            data.drop(index=condition,inplace=True)
            data['price_in_euro']=data['price_in_euro'].astype(float)
            data['year']=data['year'].astype(int)

            data_model=data.pivot_table(
            values=['power_kw','power_ps','fuel_consumption_l_100km', 'fuel_consumption_g_km']
            , index =['brand','fuel_type'], aggfunc='mean')

            for index,row in data.iterrows():
                if row['fuel_consumption_l_100km']==0:
                    data.loc[index,'fuel_consumption_l_100km']=data_model.loc[row['brand'],'fuel_consumption_l_100km'][row['fuel_type']]

            data['registration_date']=data['registration_date'].str.split('/').str[0].astype(int)
            data=data.rename(columns={'registration_date':'month'})

            return data
        
        except Exception as e:
            raise CustomException(e,sys)

        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)

            logging.info("read train and test data complete")

            target_column='price_in_euro'

            train_data=DataTransformation().cleaning_data(train_data)
            test_data=DataTransformation().cleaning_data(test_data)

            logging.info("data cleaned successfuly")

            input_feature_trian_data=train_data.drop(columns=[target_column],axis=1)
            target_feature_train_data=train_data[target_column]

            preprocesser_obj=DataTransformation().get_data_transformer_object(input_feature_trian_data)

            input_feature_test_data=test_data.drop(columns=[target_column],axis=1)
            target_feature_test_data=test_data[target_column]

            input_feature_trian_data=preprocesser_obj.fit_transform(input_feature_trian_data)
            input_feature_test_data=preprocesser_obj.transform(input_feature_test_data)

            logging.info("data trasformed successfuly")

            train_arr=np.c_[input_feature_trian_data,np.array(target_feature_train_data)]
            test_arr=np.c_[input_feature_test_data,np.array(target_feature_test_data)]

            logging.info("saved preprocessing obj")

            save_object(
                file_path=self.data_trasfromation_config.preprocessor_obj_file_path,
                obj=preprocesser_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_trasfromation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)