from open_router_model import OpenRouterGenerator
from Local_Model_generator import LocalModelGenerator


class CLI:
    def __init__(self,url_link):
        print("Loading....")
        self.initializer=OpenRouterGenerator(url_link=url_link)



    def run_app(self):
        print("---Code Generation Bot---")
        while True:
            print("\n1.Start")
            print("0.Exit")
            try:
                order = int(input("Enter your choice: "))

            except ValueError:
                print("Please enter a number.")
                continue
            if order==1:
                query=input("Enter your Query:")
                if   not query.strip():
                    print("Enter Query Correctly...")
                    continue
                print("You:\n",query)
                model_response=self.initializer.generate_code(query)
                print("Take a moment ...")
                try:
                    if  not model_response or model_response=="User Safety: safe" :
                        print("\nGeneration Failed try again later")
                        continue
                    for char in model_response:
                        print(f"\n{char}",end="",flush=True)
                    print()



                    # print("BOT:\n",model_response)


                except Exception as e:
                    print("\nGeneration Failed",str(e))


            elif order==0:
                print("GoodLuck...")
                break

            else:
                print("Enter a correctly from menu...")

if __name__=="__main__":
    app=CLI("https://openrouter.ai/api/v1/")
    app.run_app()



    
            


        