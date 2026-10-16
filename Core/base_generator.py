#import libraries and create global variables 
#Load model and tokenizer
from Core.loader import Loader
from Core.exceptions import GenerationError,ModelLoadError,ConfigurationError
from Core.config import Config
from abc import ABC , abstractmethod
from Core.Elapsed_time_func import ElapsedTimerFunction
class Generator(ABC):
    def __init__(self):
        pass




    @abstractmethod
    def generate_code(self,query:str)->str:
        pass


       


        
        