#import libraries and create global variables 
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset
from pathlib import Path
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

    def load_dataset_func(self,dataset_part:str):
            path=Path("python_data_codes_split")/ f"{dataset_part}.jsonl"
            data=load_dataset(path="json",data_files=str(path))
            return data["train"]



if __name__=="__main__":
    loader=Loader()
    train_data=loader.load_dataset_func("train")
    print(train_data["input"])


