#import libraries and create global variables 
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset

#Loader Class
class Loader:
    def __init__(self):
        self.model=None
        self.tokenizer=None
        self.dataset=None
    def load_model_and_tokenizer(self,model_name:str):
        self.model=AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer=AutoTokenizer.from_pretrained(model_name)
        return self.model , self.tokenizer

    def load_dataset(self,dataset_name:str):
        self.dataset=load_dataset(dataset_name)
        return self.dataset
    
    

