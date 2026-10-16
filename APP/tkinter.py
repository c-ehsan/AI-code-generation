import tkinter as tk 





class UI:
    def __init__(self):
        self.root=tk.Tk()
        self.root.title("CODE GENERATION APP")
        self.root.geometry("400x500")

    def create_widgets(self):
        #WELCOME LABLE
        self.welcome_lbl=tk.Label(self.root,text="Welcome To AI CODEGENERATION").pack(padx=10 ,pady=10)
        #Query LABEL
        self.client_lbl=tk.Label(self.root,text="USER:").pack(padx=10,anchor="w")
        #Query Entry
        self.client_ent=tk.Text(self.root,width=50,height=10)
        self.client_ent.pack(padx=10,pady=10,anchor="w")
        #Submit Button
        self.submit_btn=tk.Button(self.root,text="Submit").pack()
        #BOT LABEL
        self.Bot_lbl=tk.Label(self.root,text="BOT:").pack(padx=10,anchor="w")
        #BOT Entry
        self.Bot_ent=tk.Text(self.root,width=50,height=10)
        self.Bot_ent.pack(padx=10,pady=10,anchor="w")







class Logics:
    def __init__(self):
        pass

    def create_logics(self):
        pass

        


class Main:
    def __init__(self):
        self.ui=UI()
        self.ui.create_widgets()
        self.logics=Logics()
        self.ui.Bot_ent.configure(state="disabled")


    def run(self):
        self.ui.root.mainloop()
    


if __name__=="__main__":
    main=Main()
    main.run()



