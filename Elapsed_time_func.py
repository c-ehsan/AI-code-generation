import time 

class ElapsedTimerFunction:
    def __call__(self, func):
        def wrapper(*args,**kwargs):
            start_time=time.perf_counter()
            for chunk in func(*args , **kwargs):
                yield chunk

            end_time=time.perf_counter()
            print(f"Elapsed timer result: {end_time - start_time} seconds")

        return wrapper
        