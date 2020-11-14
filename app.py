from flask import Flask, request, jsonify
import flask
import joblib
import pandas as pd

app = Flask(__name__, template_folder='templates')


# Use joblib to load in the pre-trained model
with open(f'model/cat_finalized_model.pkl', 'rb') as f:
    model = joblib.load(f)

# Default Flask API Gateway
@app.route('/', methods=['GET', 'POST'])
def default_gateway():
    ## if we receive a GET request, render the homepage or the index.html page
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))

    ## when user submits prediction request
    if flask.request.method == 'POST':

        ## extract values from the form object
        climate_vs = flask.request.form['climate_vs']
        climate_def = flask.request.form['climate_def']
        climate_vap = flask.request.form['climate_vap']
        climate_aet = flask.request.form['climate_aet']
        precipitation = flask.request.form['precipitation']
        landcover_5 = flask.request.form['landcover_5']

        ## create a pandas DataFrame from this data
        predictor_variables = pd.DataFrame([[climate_vs, climate_def, climate_vap, climate_aet, precipitation, landcover_5]],
            columns=['climate_vs', 'climate_def', 'climate_vap', 'climate_aet', 'precipitation', 'landcover_5'], dtype=float)

        ##make predictions on the model
        predicted_burned_area = model.predict(predictor_variables)[0]

        ## return the prediction to the client

        return flask.render_template('index.html', 
            original_input={'Climate_Vs':climate_vs,'Climate_Def':climate_def, 'Climate_Vap':climate_vap,
            'Climate_Aet':climate_aet,'Precipitation':precipitation, 'Landcover':landcover_5}, result= round(predicted_burned_area, 3))



if __name__ == '__main__':
    app.run(port=5001)