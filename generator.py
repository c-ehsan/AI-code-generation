#import libraries and create global variables 
#Load model and tokenizer
from loader import Loader
from exceptions import GenerationError,ModelLoadError,ConfigurationError
from config import Config

class Generator:
    def __init__(self):
        try:
            config=Config()
            config.load_config()
            self.model_name=config.get_model_name()
            self.model , self.tokenizer=Loader().load_model_and_tokenizer(self.model_name)

        except Exception as e:
            raise ModelLoadError("Failed to load model...") from e 
        


    def generate_code(self,query):
        try:
            inputs=self.tokenizer(query,return_tensors="pt")
            model_response=self.model.generate(**inputs,max_new_tokens=200)
            outputs=self.tokenizer.decode(model_response[0])
            print("Model_response:\n",end="",flush=True)
            for item in outputs:
                print(item,end="",flush=True)
        except Exception as e:
            raise GenerationError("Failed to Generate output...") from e



generator=Generator()
generator.generate_code("generate a python loop")



        
        