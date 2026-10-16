import json 
import pandas as pd 
class Config:
    def load_config(self):
        with open("config.json","r") as f:
            self.__loaded_data=json.load(f)
            self.model_name=self.__loaded_data["model_name"][0]
            self.dataset_name=self.__loaded_data["dataset_name"][0]
            return {"model_name":self.model_name , "dataset_name":self.dataset_name}



