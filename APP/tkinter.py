
import tkinter as tk
import threading
import queue

from Core.open_router_model import OpenRouterGenerator


class UI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Code Generation")
        self.root.geometry("500x600")

    def create_widgets(self):
        self.welcome_lbl = tk.Label(
            self.root, text="Welcome to AI Code Generation"
        )
        self.welcome_lbl.pack(pady=10)

        self.client_lbl = tk.Label(self.root, text="USER:")
        self.client_lbl.pack(anchor="w", padx=10)

        self.client_ent = tk.Text(self.root, width=55, height=8)
        self.client_ent.pack(padx=10, pady=5)

        self.submit_btn = tk.Button(self.root, text="Submit")
        self.submit_btn.pack(pady=10)

        self.bot_lbl = tk.Label(self.root, text="BOT:")
        self.bot_lbl.pack(anchor="w", padx=10)

        self.bot_ent = tk.Text(
            self.root, width=55, height=15, state="disabled"
        )
        self.bot_ent.pack(padx=10, pady=5)


class Logics:
    def __init__(self, url_link):
        self.model_initializer = OpenRouterGenerator(url_link)

    def return_model_response(self, query):
        return self.model_initializer.generate_code(query)


class Main:
    def __init__(self):
        self.ui = UI()
        self.ui.create_widgets()

        self.logics = Logics("https://openrouter.ai/api/v1/")
        self.messages = queue.Queue()

        self.ui.submit_btn.configure(command=self.generate)
        self.ui.root.after(100, self.process_queue)

    def generate(self):
        query = self.ui.client_ent.get("1.0", tk.END).strip()

        if not query:
            self.show_response("Please enter a query.")
            return

        self.ui.submit_btn.configure(
            state="disabled", text="Generating..."
        )
        self.show_response("")

        # Run the API request in a separate thread
        thread = threading.Thread(
            target=self.worker,
            args=(query,),
            daemon=True
        )
        thread.start()

    def worker(self, query):
        try:
            # generate_code uses yield, so iterate over its output
            for chunk in self.logics.return_model_response(query):
                if chunk:
                    self.messages.put(("chunk", chunk))

            self.messages.put(("done", None))

        except Exception as e:
            self.messages.put(("error", str(e)))

    def process_queue(self):
        try:
            while True:
                message_type, content = self.messages.get_nowait()

                if message_type == "chunk":
                    self.ui.bot_ent.configure(state="normal")
                    self.ui.bot_ent.insert(tk.END, content)
                    self.ui.bot_ent.see(tk.END)
                    self.ui.bot_ent.configure(state="disabled")

                elif message_type == "done":
                    self.ui.submit_btn.configure(
                        state="normal", text="Submit"
                    )

                elif message_type == "error":
                    self.show_response(f"Error: {content}")
                    self.ui.submit_btn.configure(
                        state="normal", text="Submit"
                    )

        except queue.Empty:
            pass

        # Check the queue again after 100 milliseconds
        self.ui.root.after(100, self.process_queue)

    def show_response(self, text):
        self.ui.bot_ent.configure(state="normal")
        self.ui.bot_ent.delete("1.0", tk.END)
        self.ui.bot_ent.insert(tk.END, text)
        self.ui.bot_ent.configure(state="disabled")

    def run(self):
        self.ui.root.mainloop()


if __name__ == "__main__":
    app = Main()
    app.run()
