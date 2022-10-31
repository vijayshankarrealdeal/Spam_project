from src import data_preprocess
from src import train
from src import run
import matplotlib.pyplot as plt
from ui import main_ui

Train = input("Enter True to train or False to play results.")
split_size = 5000
if Train == "True":
    history = run.run_train(Train)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.show()
else:
    main_ui.make_ui()



