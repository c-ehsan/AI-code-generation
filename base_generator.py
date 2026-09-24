#import libraries and create global variables 
#Load model and tokenizer
from loader import Loader
from exceptions import GenerationError,ModelLoadError,ConfigurationError
from config import Config
from abc import ABC , abstractmethod
from Elapsed_time_func import ElapsedTimerFunction
class Generator(ABC):
    def __init__(self):
        pass




    @abstractmethod
    def generate_code(self,query:str)->str:
        pass


       


        
        