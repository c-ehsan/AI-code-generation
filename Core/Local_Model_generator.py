from Core.base_generator import Generator
from Core.Elapsed_time_func import ElapsedTimerFunction
elapsed_timer_func=ElapsedTimerFunction()
class LocalModelGenerator(Generator):
    def __init__(self,model_path:str):
        pass


        
    @elapsed_timer_func
    def generate_code(self, query)->str:
        pass


if __name__=="__main__":
    localmodel=LocalModelGenerator()
    localmodel.generate_code()