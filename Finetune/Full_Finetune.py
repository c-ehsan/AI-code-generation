#import libraries and create global variabels
from transformers import  Trainer , TrainingArguments , GenerationConfig,DataCollatorForLanguageModeling, EarlyStoppingCallback
from peft import PeftModel , get_peft_model  , LoraConfig
from Core.loader import Loader
from Core.config import Config
from .base_Finetune import BaseFineTune

#FullFineTune class
class FullFinetune(BaseFineTune):
    def __init__(self):
        initializer=Loader()
        self.config=Config().load_config()
        self.data_loader=Loader()
        self.model , self.tokenizer=initializer.load_model_and_tokenizer(model_name=self.config["model_name"])
        self.train_dataset=self.data_loader.load_dataset_func("train")
        self.test_dataset=self.data_loader.load_dataset_func("test")


    def tokenizer_func(self,example):
        start_prompt="Generate Code the following text\n\n"
        end_prompt="\n\nGeneration:"
        prompt=[start_prompt +input_text + end_prompt for input_text in example["input"]]
        example["input_ids"]=self.tokenizer(prompt,padding=True, max_length=128 , truncation=True , return_tensors="pt").input_ids

        example["labels"]=self.tokenizer(example["output"],padding=True, max_length=128 , truncation=True , return_tensors="pt" ).input_ids

        return example
                



    def tokenized_dataset_func(self,data):
        self.tokenized_dataset=data.map(self.tokenizer_func,batched=True)
        self.tokenized_dataset=self.tokenized_dataset.remove_columns(data.column_names)
        return self.tokenized_dataset
    def datacollator_func(self):
        self.data_collator=DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer ,mlm=False
        )


    def training_args_func(self,per_device_train_batch_size:int ,per_device_eval_batch_size:int,epochs:int,learning_rate:int ,logging_steps:int,**kwargs):
          self.training_args=TrainingArguments(
              output_dir="model_config",
              per_device_train_batch_size=per_device_train_batch_size,
              per_device_eval_batch_size=per_device_eval_batch_size,
              num_train_epochs=epochs,
              learning_rate=learning_rate,
              logging_steps=logging_steps,
              max_steps=-1,
              **kwargs
          )
  



    def trainer_func(self):
        self.trainer=Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.data_collator,
            train_dataset=self.tokenized_dataset_func(self.train_dataset),
            eval_dataset=self.tokenized_dataset_func(self.test_dataset),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]


        )
        self.trainer.train()

  
    def run(self):
        self.datacollator_func()
        self.training_args_func(
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            epochs=3,
            learning_rate=0.0002,
            logging_steps=50,
            gradient_accumulation_steps=2,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            load_best_model_at_end=True,
            remove_unused_columns=False,
            save_strategy="epoch",
            warmup_steps=0.01,
            eval_strategy="epoch",
            save_steps=50,
            eval_steps=50
        )
        self.trainer_func()
        self.save_model_func()


if __name__=="__main__":
    fullfintune=FullFinetune()
    fullfintune.run()
