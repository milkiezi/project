from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import onnxruntime as rt
import json

app = Flask(__name__)


@app.route('/predict',methods=['GET','POST'])
def home():
    sess = rt.InferenceSession('pc_game_with_rating_model.ort')
    user_input = request.form.get('input')
    #u = [user_input]
    #print(user_input,type(user_input))
    #print(u,type(u))
    #res = [float(value) for value in u.split(', ')]
    #data = list(map(float, u.split(',')))
    #print(res)
    l = []
    l.extend(map(int, user_input.split(",")))
    print(l)
    #a = np.asarray(json.loads(user_input), dtype=np.float32)
    #print(a)
    data = np.array([l])
    #print('data=',data,type(data))
    #data = np.array([[33,67902,2419,691,402,7,2004,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
    probabilities = sess.run(['label'],
                             {'features': data.astype(np.float32)})
         
    #print('prob=',probabilities)
    return render_template('index.html',probabilities=probabilities)
    #return render_template('index.html')




if __name__ == '__main__':
    app.run(debug=True)