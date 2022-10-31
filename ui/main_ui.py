import tkinter as tk
from src import test
import uuid
from tkinter.filedialog import askopenfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def make_ui():
    root= tk.Tk()
    root.title('Spam Filter')
    canvas1 = tk.Canvas(root, width = 600, height = 400, bg='#191919')
    canvas1.pack()

    entry1 = tk.Entry(root,width=70) 
    entry1.insert(0, 'enter your text.')
    entry1.pack()

    entry1.place(height = 100)
    canvas1.create_window(300, 140, window=entry1)

    def check_spam_res():
        x1 = entry1.get()
        prediction =  test.test(x1)
        if len(x1) != 0:
            result = "Spam found" if np.round(prediction) > 0.7 else "Good no spam found"
            label1 = tk.Label(root, text =result)
        else:
            label1 = tk.Label(root,text="add some text")
        canvas1.create_window(300, 230, window=label1)
    
    def add_txt_file():
        label2 = tk.Label(root,text="Wait....")
        canvas1.create_window(300, 330, window= label2)
        file = askopenfile(mode ='r', filetypes =[('Text Files', '*.txt')])
        if file is not None:
            
            content = [i.strip() for i in file.readlines()]
            df = pd.DataFrame(content,columns=['text'])
            res = df.text.apply(lambda x :  test.test(x).squeeze())
            df['result'] = res.apply(lambda x : "Spam found" if np.round(x) > 0.7 else "Good no spam found")
            fig, ax = plt.subplots()
            fig.patch.set_visible(False)
            ax.axis('off')
            ax.axis('tight')
            df.text = df.text.apply(lambda x : x[:8] + '.....')
            ax.table(cellText=df.values, colLabels=df.columns, loc='center')
            fig.tight_layout()
            file_name = str(uuid.uuid4())
            df.to_csv(f'./out/{file_name}.csv',index= False)
            plt.title(f"Saved as {file_name} in out folder.",fontsize = 6,loc='center')
            plt.show()
        root.destroy()
        
    button1 = tk.Button(text=' Check Spam ',height= 1,width=20, command=check_spam_res)
    button2 = tk.Button(text=' Upload Text File ',height= 1,width=20, command=add_txt_file)
    canvas1.create_window(300, 180, window=button1)
    canvas1.create_window(300, 280, window=button2)
    root.mainloop()

