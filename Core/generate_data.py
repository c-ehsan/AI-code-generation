#import libraries and create global variabels
from datasets import load_dataset , Dataset
import pandas as pd
from pathlib import Path


class GenerateData:
    SEED=42
    OUTPUT_DIR=Path("python_data_codes_split")
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

    def __init__(self,dataset_name:str):
        self.dataset=load_dataset(dataset_name , split="train")



    def management_data(self):
        df=self.dataset.to_pandas()
        required_cols=['instruction', 'output', 'input']
        missing_cols=set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing Columns:{missing_cols}")

        #dropna-->remove missing values or NANS
        df=df.dropna(subset=["instruction","input","output"]).copy()

        #normalize_data 
        for column in required_cols:
            df[column]=df[column].fillna("").astype("str").str.strip()

        df=df = df[
    (df["instruction"] != "")
    & (df["output"] != "")
].copy()


        df=df.drop_duplicates(subset=["instruction", "input", "output"]).reset_index()


        self.clean_dataset=Dataset.from_pandas(df,preserve_index=False)




    def generate(self):
        first_split=self.clean_dataset.train_test_split(test_size=0.20,seed=self.SEED,shuffle=True)
        second_split=first_split["test"].train_test_split(test_size=0.50,seed=self.SEED,shuffle=True)


        self.train_dataset=first_split["train"]
        self.validation_dataset=second_split["train"]
        self.test_dataset=second_split["test"]




    def convert_to_json(self):
        splits={
            "train":self.train_dataset,
            "validation":self.validation_dataset,
            "test":self.test_dataset

        }
        for split_name  , split_data in splits.items():
            output_path=self.OUTPUT_DIR/f"{split_name}.jsonl"

            split_data.to_json(
                str(output_path),
                orient="records",
                lines=True,
                force_ascii=False
            )

            print(f"{split_name} : {len(split_data)} rows")
            print(f"Saved as:{output_path}")
    @classmethod
    def run(cls):
        cls.management_data()
        cls.generate()
        cls.convert_to_json()


if __name__=="__main__":
    generatedata=GenerateData("flytech/python-codes-25k")