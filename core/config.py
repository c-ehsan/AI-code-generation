import json 
import pandas as pd 
class Config:
    def load_config(self):
        with open("config.json","r") as f:
            self.__loaded_data=json.load(f)
            self.model_name=self.__loaded_data["model_name"][0]
            self.dataset_name=self.__loaded_data["dataset_name"][0]

    def get_model_name(self):
        return self.model_name

    def get_dataset_name(self):
        return self.dataset_name


