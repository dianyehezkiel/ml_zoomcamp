import pickle
from flask import Flask
from flask import request
from flask import jsonify

with open('model.bin', 'rb') as model_bin, open('dv.bin', 'rb') as dv_bin:
  model = pickle.load(model_bin)
  dv = pickle.load(dv_bin)

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]

    result = {
        'churn_probability': float(round(y_pred, 3)),
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)