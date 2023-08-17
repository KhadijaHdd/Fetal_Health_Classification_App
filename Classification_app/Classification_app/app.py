from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler_model = pickle.load(open('scaler_model.pkl', 'rb'))
@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    normalized_features = scaler_model.transform(final_features)
    prediction = model.predict(normalized_features)
    output = round(prediction[0], 2)
    if output == 1.0:
        prediction_class = "Normal"
    elif output == 2.0:
        prediction_class = "Suspect"
    elif output == 3.0:
        prediction_class = "Pathologique"

    return render_template('index.html', prediction_text=" est   {}".format(prediction_class))

if __name__ == "__main__":
    app.run(debug=True)