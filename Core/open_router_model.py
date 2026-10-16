from Core.base_generator import Generator
import os 
from Core.exceptions import APINotFoundError
from openai import OpenAI
from dotenv import load_dotenv
from Core.Elapsed_time_func import ElapsedTimerFunction
from Core.prompt_builder import PromptBuilder
load_dotenv()
class OpenRouterGenerator(Generator):
    
    def __init__(self,url_link):
        self.__url_link=url_link
        api_key=os.getenv("OPENROUTER_API_KEY")
        try:

            self.client=OpenAI(
                base_url=self.__url_link,
                api_key=api_key
            )

        except Exception as e:
             raise APINotFoundError("Api key  was not found.") from e 
        
    

    @ElapsedTimerFunction()
    def generate_code(self, query:str):

        try:
            response=self.client.chat.completions.create(
                model="openrouter/free",
                messages=PromptBuilder.build(query),
                max_tokens=200,
                stream=True

            )

            for chunk in response:
                content=chunk.choices[0].delta.content

                if  not chunk.choices:
                    print("I cant understandGive another query")
                    continue

                if content:
                    yield content


        except Exception as e:
            raise APINotFoundError("Failed to generate code using OpenRouter.") from e 







        