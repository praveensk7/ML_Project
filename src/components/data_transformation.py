import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
sys.path.append('src/')
from exception import CustomException
sys.path.append('src/')
from logger import logging
sys.path.append('src/')
from utils import save_object




@dataclass
class DataTransformationConfig:
    preprocessor_ob_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        try:
            numerical_columns=["writing_score","reading_score"]
            categorical_columns=['gender',
                                'race_ethnicity',
                                'parental_level_of_education',
                                'lunch',
                                'test_preparation_course']
            
            num_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("onehotencoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info("pipelines created for num and cat")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            
            logging.info("transformer created for num and cat")

            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformer(self,train_path,test_path):
        try:
            df_train=pd.read_csv(train_path)
            df_test=pd.read_csv(test_path)

            logging.info("train and test read")

            preprocessor_obj=self.get_data_transformer_obj()
            target_column=["math_score"]

            numerical_columns=["writing_score","reading_score"]
            input_feature_train_df=df_train.drop(columns=target_column,axis=1)
            target_feature_train_df=df_train[target_column]

            input_feature_test_df=df_test.drop(columns=target_column,axis=1)
            target_feature_test_df=df_test[target_column]
            sc=StandardScaler()

            
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_path,
                obj=preprocessor_obj
            )

            return (train_arr,test_arr,self.data_transformation_config.preprocessor_ob_path)

        except Exception as e:
            raise CustomException(e,sys)