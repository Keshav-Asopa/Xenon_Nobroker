#importing libraries
import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/home')
def home():
    return flask.render_template('home.html')


#prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,38)
    print(to_predict)
    loaded_model = pickle.load(open("model.pkl","rb"))
    print(loaded_model)
    result = loaded_model.predict(to_predict)
    print(result)
    return result[0]


@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        print(to_predict_list)
        to_predict_list=list(to_predict_list.values())
        print(to_predict_list)
        to_predict_list = list(map(int, to_predict_list))
        print(to_predict_list)
        result = ValuePredictor(to_predict_list)
        return str(result)
        # if 1:
        #     return str(result)
        # else:
        #     return 'Income less than 50K'

if __name__=='__main__':
    #port=int(os.environ.get('PORT',5000))
	app.run(port=5000,host='0.0.0.0',debug=True)