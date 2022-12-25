from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
from flask.json import jsonify
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
 
import warnings
from joblib import dump, load
from sklearn.linear_model import LogisticRegressionCV



app = Flask(__name__)

model= load( 'model.pkl' )
#model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template("forest_fire.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('forest_fire.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),bhai="kuch karna hain iska ab?")
    else:
        return render_template('forest_fire.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),bhai="Your Forest is Safe for now")

@app.route("/test", methods = ["GET","POST"])
def fetchLine():
    data = request.get_json()  
    print(data)
    print(request.json)
    print(':: => ',request.args)
    data = [request.args.get("hr"),request.args.get("age"),request.args.get("status")  ]
  
    int_features=[int(x) for x in data]
    
    inF  = []
    inF.append((int_features[0]-50)/(120-50))
    inF.append((int_features[1]-10)/(110-10))
    tp = [0,0,0,0]
    tp[int_features[-1]-1] =1
    inF.extend(tp)
 
    final= np.array(inF) 
 
    print(final)
    output=model.predict(final.reshape(1,-1))
 
    return  jsonify({'output':str(output)} )         
    

if __name__ == '__main__':

    app.run(debug=True)
