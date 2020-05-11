from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('suv.pkl','rb'))

@app.route('/')
def hey():
    return render_template('suv.html')
@app.route('/predict', methods=['POST','GET'])
def predict():

    val = [int(x) for x in request.form.values()]
    res = [np.array(val)]
    pred = model.predict(res)

    #return render_template('suv.html',ans='res {}'.format(pred))

    if pred >= 1:
        return render_template('suv.html',atext="This Guy must have a SUV")
    else:
        return render_template('suv.html',atext="This Guy Should Not have a SUV")

# @app.route('/test', methods=['POST','GET'])
# def test():
#     post_title = request.form['Age']
#     return render_template('suv.html',text=post_title)



if __name__ == '__main__':
    app.run(debug=True)
