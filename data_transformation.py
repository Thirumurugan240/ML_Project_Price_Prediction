import os
import pandas as pd
import numpy as np
import sys
from dataclasses import dataclass
from src.my_logging import logging
from src.exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from src.utils import save_object

@dataclass
class datatranformationconfig:
    preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

class datatranformation:
    def __init__(self):
        self.data_tranformation_config = datatranformationconfig()

    def get_tranform_data(self):
        try:
            num_features = ['Trip_Distance_km', 'Per_Km_Rate', 'Per_Minute_Rate', 'Trip_Duration_Minutes']
            cat_features = ['Time_of_Day', 'Day_of_Week', 'Traffic_Conditions', 'Weather']
            target_feature = ['Trip_Price']

            # Create the numerical pipeline
            num_preprocessor = Pipeline(
                steps=[
                    ('SimpleImputer', SimpleImputer(strategy='median')),
                    ('StandardScaler', StandardScaler())
                ]
            )

            # Create the categorical pipeline
            cat_preprocessor = Pipeline(
                steps=[
                    ('SimpleImputer', SimpleImputer(strategy='most_frequent')),
                    ('OneHotEncoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                    ('StandardScaler', StandardScaler())
                ]
            )

            target_preprocessor = Pipeline(
                steps=[
                    ('SimpleImputer', SimpleImputer(strategy='median'))
                ]
            )

            # Combine both pipelines in a ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_preprocessor, num_features),
                    ('cat_pipeline', cat_preprocessor, cat_features),
                    ('target_preprocessor',target_preprocessor,target_feature)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_tranformation(self, train_data, test_data):
        try:
            # Read train and test data
            train_data = pd.read_csv(train_data)
            test_data = pd.read_csv(test_data)

            logging.info('Read train and test data completed')

            # Obtain the preprocessing object
            logging.info('Obtaining preprocessing object')
            preprocessor_obj = self.get_tranform_data()

            # Separate features and target column
            #target_column = 'Trip_Price'
            #input_train_data = train_data.drop(columns=target_column, axis=1)
            #target_train_data = train_data[target_column]
            #input_test_data = test_data.drop(columns=target_column, axis=1)
            #target_test_data = test_data[target_column]


            # Apply transformations to the training and test data
            preprocessed_train_input = preprocessor_obj.fit_transform(train_data)
            preprocessed_test_input = preprocessor_obj.transform(test_data)  # Use transform for test data

            input_train_data = preprocessed_train_input[:,:-1]
            target_train_data = preprocessed_train_input[:,-1]
            input_test_data = preprocessed_test_input[:,:-1]
            target_test_data = preprocessed_test_input[:,-1]

            # Combine transformed input with target variable
            train_arr = np.c_[input_train_data, np.array(target_train_data)]
            test_arr = np.c_[input_test_data, np.array(target_test_data)]

            # Save the preprocessor object
            save_object(
                file_path=self.data_tranformation_config.preprocessor_path,
                obj=preprocessor_obj
            )
            logging.info('Preprocessing_completed')

            return (train_arr, test_arr)
        except Exception as e:
            
            raise CustomException(e, sys)
