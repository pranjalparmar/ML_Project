import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_file_path: str=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''This function is used to create a data transformation pipeline which will be used to transform the data before feeding it to the model'''

        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_colmuns = ["gender", 
                                   "race_ethnicity", 
                                   "parental_level_of_education", 
                                   "lunch", 
                                   "test_preparation_course"]
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(sparse_output=False)),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Data Transformation Pipeline Created Successfully: Categorical Columns:{categorical_colmuns} & Numerical Columns{numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num", num_pipeline, numerical_columns),
                    ("cat", cat_pipeline, categorical_colmuns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        '''This function is used to initiate the data transformation process and save the preprocessor object as a pickle file'''
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Train & Test data Ingested Successfully as DataFrame") 

            logging.info("Obtaining preprocessing object")

            preprocessing_object=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying Preprocessing object on training df and testing df"
            )

            input_feature_train_array=preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_array=preprocessing_object.transform(input_feature_test_df)

            train_array = np.c_[input_feature_train_array, np.array(target_feature_train_df)]
            test_array = np.c_[input_feature_test_array, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing Object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessing_object
            )

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_file_path
            )
        


        except Exception as e:
            logging.error("Error in Data Transformation")
            raise CustomException(error_message=e, error_detail=sys)