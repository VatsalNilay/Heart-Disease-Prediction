from flask import render_template, request, Flask
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)


filename = "model/finalized_model.sav"
loaded_model = pickle.load(open(filename, "rb"))


data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# change the input data to a numpy array
data_array = np.asarray(data)

# reshape the numpy array as we are predicting for only on instance
reshaped_data = data_array.reshape(1,-1)


result1 = loaded_model.predict(reshaped_data)

answer = result1[0]

@app.route("/")
def home():
    return render_template("index.html",value = answer)

if __name__ == "__main__":
    app.debug = True
    app.run()