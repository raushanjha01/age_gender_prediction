from distutils.command.upload import upload
#from pyspark.ml.pipeline import PipelineModel
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, Response
import os
import pickle
import pandas as pd

app = Flask(__name__)

UPLOAD_FOLDER = "static/files"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

age_model = pickle.load(
    open('age_model.pkl', 'rb'))

ap_data_model = pickle.load(
    open('ap_data_model.pkl', 'rb'))

#gender_model = PipelineModel.load("gender_predict_model/")
#gp_data_model = PipelineModel.load("gp_data_model/")


@app.route('/', methods=['POST','GET'])
def home():
    return render_template('index.html')

# predict the age of the device owner using the model
def predict_campaign_by_age(df):
    transformed_df = ap_data_model.transform(df)
    age = round(age_model.predict(transformed_df)[0],2)
    return choose_campaign_age(age)
'''
def predict_gender(df):
    transformed_DF = gp_data_model.transform(df)
    predictions = gender_model.transform(transformed_DF)
    return predictions
'''

def  choose_campaign_age(age):
    campaign = 'Campaign 5 - Special Offers for payment wallet offers.'
    # calculate age group
    if (age <= 24):
        campaign = 'Campaign 4 - Bundled smartphone offers.'
    elif (age >= 32):
        campaign = 'Campaign 6 - Special cashback offers for Priviliged Membership.'

    return campaign

@app.route('/getCampaign', methods=['POST'])
def getCampaign():
    '''
    For rendering results on HTML GUI
    '''
    longitude = float(request.form.get('longitude'))
    latitude = float(request.form.get('latitude'))
    Day = request.form.get('Day')
    hour = int(request.form.get('Hour'))
    week = int(request.form.get('week'))
    '''app_id = int(request.form.get('app_id'))
    label_id = int(request.form.get('label_id'))
    category = int(request.form.get('category'))'''
    
    data = [[longitude,latitude,Day, hour, week]]
    df = pd.DataFrame(data, columns=['longitude', 'latitude', 'Day', 'Hour', 'week'])
    campaign = predict_campaign_by_age(df)

    '''data = [[longitude,latitude,app_id, label_id, category, Day, hour]]
    df = pd.DataFrame(data, columns=['longitude', 'latitude', 'app_id', 'label_id','category', 'Day', 'Hour'])
    gender_prediction = predict_gender(df)
    print(gender_prediction)'''

    return render_template('index.html', prediction_text='The suitable campaign is {}'.format(campaign))


def find_campaign(filepath, given_file_name):
    csvdata = pd.read_csv(filepath)
    print(csvdata['longitude'])
    csvdata = csvdata.dropna()
    csvdata = csvdata.filter(['longitude','latitude','Day','Hour','week'], axis=1)
    csvdata = ap_data_model.transform(csvdata)
    csvdata["predicted_age"] = round(age_model.predict(csvdata)[0],2)
    csvdata['campaign'] = csvdata.predicted_age.apply(lambda x:choose_campaign_age(x))
    print (csvdata)
    predicted_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_' + given_file_name)
    csvdata.to_csv(predicted_filepath)
    return predicted_filepath

@app.route("/getGetCSV")
def getGetCSV():
    filename = request.args.get("given_file")
    predicted_file = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_' + filename)
    csv = pd.read_csv(predicted_file)
    return Response(
        csv.to_csv(index=False),
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=predicted_" + filename})

@app.route('/uploadFiles', methods=['POST'])
def uploadFiles():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)
        print(file_path)
        predicted_file = find_campaign(file_path, uploaded_file.filename)
    #return render_template('index.html', prediction_text='File uploaded. <a href="/getGetCSV?given_file=' + uploaded_file.filename + '">Click here to download predicted file.</a>')
    return render_template('download.html', file_name = uploaded_file.filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port='8080')


