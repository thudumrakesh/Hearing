from flask import Flask,render_template,url_for,request,redirect
import numpy as np
import pandas as pd
import joblib
import pickle


app = Flask(__name__)

model = joblib.load('model.pkl')
scale = joblib.load('scaler_joblib')


@app.route('/')
@app.route('/main')
def main():
	return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
	int_features =[[x for x in request.form.values()]]
	c = ["age","physical_score"]
	final = pd.DataFrame(int_features,columns=c)
	result = model.predict(final)
	print("The Result is :",result)
	if result == 1:
		hear='pass'
	else :
		hear="fail"


	print(int_features)

	return render_template("main.html",prediction_text="The predicated lab record is : {}".format(hear))


if __name__ == "__main__":
	app.debug=True
	app.run(host='0.0.0.0', port=7000)
