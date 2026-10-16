#import libraries and create global variabels
from abc import ABC , abstractmethod
from transformers import  Trainer , TrainingArguments , GenerationConfig,DataCollatorForLanguageModeling, EarlyStoppingCallback
from peft import PeftModel , get_peft_model  , LoraConfig
from Core.loader import Loader
from Core.config import Config
#Base Class 

class BaseFineTune(ABC):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def tokenizer_func(self):
        pass
    @abstractmethod
    def tokenized_dataset_func(self):
        pass

    def datacollator_func(self):
        pass
    @abstractmethod
    def training_args_func(self):
        pass
    @abstractmethod
    def trainer_func(self):
        pass

    @abstractmethod
    def run():
        pass




class Peft(BaseFineTune):
    def __init__(self):
        initializer=Loader()
        self.model , self.tokenizer=initializer.load_model_and_tokenizer(model_name=self.config.get_model_name())
        self.data=initializer.load_dataset(dataset_name=self.config.get_dataset_name())
        self.train_data=self.data["train"]
        self.input_text=self.train_data["input"]
        self.test_data=self.train_data*0.2

    def tokenizer_func(self,example):
        start_prompt="Generate Code the following text\n\n"
        end_prompt="\n\nGeneration:"
        prompt=[start_prompt +input_text + end_prompt for input_text in example["input"]]
        example["input_ids"]=self.tokenizer(prompt,padding="max_length" , max_length=128 , trunction=True , return_tensors="pt").input_ids

        example["labels"]=self.tokenizer(example["output"],padding="max_length" , max_length=128 , trunction=True , return_tensors="pt" ).input_ids

        return example
        


    def tokenized_dataset_func(self):
        self.tokenized_dataset=self.data.map(self.tokenizer_func(self.input_text))
        self.tokenized_dataset=self.data.remove_columns(["instruction", "text"])

    def datacollator_func(self):
        self.data_collator=DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer , model=self.model 
        )


    def build_lora_config(self):
        self.lora_config=LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CASUAL_LM"


        )

    def adaptation_lora(self):
        self.peft_model=PeftModel(self.model,self.lora_config)
        self.peft_model.print_trainable_parameters()

    


    def training_args_func(self,per_device_train_batch_size:int ,per_device_eval_batch_size:int,epochs:int , gradient_accumulation_steps:int,learning_rate:int ,logging_steps:int,weigth_decay:float):
        self.training_args=TrainingArguments(
            output_dir="model_config",
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            epochs=epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            weight_decay=weigth_decay,
            lr_scheduler_type="cosine",
            max_steps=20,
            warmup_steps=0.01, 
            load_best_model_at_end=True
        )



    def trainer_func(self):
        self.trainer=Trainer(
            model=self.peft_model,
            args=self.training_args,
            data_collator=self.data_collator,
            train_dataset=self.dataset["train"],
            eval_dataset=self.test_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]


        )
        self.trainer.train()


    def eval_trainer(self):
        pass

    def save_model_func(self):
        self.peft_model.save_pretrained("./Peft_model_configuration",from_pt=True)


    @classmethod
    def run(cls):
        cls.tokenized_dataset_func()
        cls.datacollator_func()
        cls.build_lora_config()
        cls.adaptation_lora()
        cls.training_args_func(per_device_train_batch_size=2,per_device_eval_batch_size=2,epochs=3,gradient_accumulation_steps=2,learning_rate=0.0002,logging_steps=50,weight_decay=0.01)
        cls.trainer_func()
        cls.save_model_func()


if __name__=="__main__":
    peft=Peft()




        


