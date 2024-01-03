import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
knn_model=pickle.load(open('final_model_knn.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=knn_model.predict(new_data)
    print(output[0])
    if output == 0:
        output_class = 'Adelie'
    elif output == 1:
        output_class = 'Chinstrap'
    else:
        output_class = 'Gentoo' 
    serialized = int(output[0])
    return jsonify(output_class)

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input.reshape(1,-1))
    output=knn_model.predict(final_input)[0]
    if output == 0:
        output_class = 'Adelie'
    elif output == 1:
        output_class = 'Chinstrap'
    else:
        output_class = 'Gentoo' 
    return render_template("home.html",prediction_text="The Penguins belongs to the {} species".format(output_class))



if __name__=="__main__":
    app.run(debug=True)
   
     