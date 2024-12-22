import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.my_logging import logging
from src.exception import CustomException
from src.components.model_trainer import Model_trainer
from src.components.data_transformation import datatranformation
from dataclasses import dataclass


@dataclass
class dataingestconfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','data.csv')

class dataingestion():
    def __init__(self):
        self.data_config_ingest=dataingestconfig()

    def initiate_data_ingestion(self):
        try:
            logging.info('Entered data ingestion')
            df = pd.read_csv('dataset/taxi_trip_pricing.csv')
            df.to_csv(self.data_config_ingest.raw_data_path,header=True,index=False)

            os.makedirs(os.path.dirname(self.data_config_ingest.train_data_path),exist_ok=True)

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.data_config_ingest.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_config_ingest.test_data_path,index=False,header=True)

            logging.info('Data splitting completed')

            return(
                self.data_config_ingest.train_data_path,
                self.data_config_ingest.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    obj = dataingestion()
    train_data,test_data = obj.initiate_data_ingestion()

    obj_2 = datatranformation()
    train_array,test_array = obj_2.initiate_data_tranformation(train_data,test_data)

    modeltrainer = Model_trainer()
    print(modeltrainer.initiate_model_trainer(train_array,test_array))



