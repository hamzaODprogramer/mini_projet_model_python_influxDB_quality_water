from flask import Flask,render_template,url_for,request,redirect,jsonify
from influxDB import InfluxDB
import pandas as pd
import joblib
import os
import yaml

def load_config():
  with open('config.yaml', 'r') as file:
    return yaml.safe_load(file)


app = Flask(__name__)

config = load_config()
infexdb = InfluxDB(config["token"], config["org"], config["url"], config["bucket"], config["measurement"], 'NULL' ,0)
model_filename = 'decision_tree_model.joblib'
clf = joblib.load(model_filename)

@app.route('/')
def index():
 return render_template('index.html')

@app.route('/forme')
def forme():
 return render_template('forme.html')

@app.route('/get_result',methods=['POST','GET'])
def evaluate():
    new_data = pd.DataFrame({
        'Residual Free Chlorine (mg/L)': [request.form['val1']],
        'Turbidity (NTU)': [request.form['val2']],
        'Fluoride (mg/L)': [request.form['val3']],
        'Coliform (Quanti-Tray) (MPN /100mL)': [request.form['val4']],
        'E.coli(Quanti-Tray) (MPN/100mL)': [request.form['val5']]
    })
    data = clf.predict(new_data)

    # Convert the NumPy array to a string
    result_string = str(data)

    return render_template('forme.html', data=result_string)


@app.route('/dashboard')
def admin():
  return render_template('admin.html')


@app.route('/login')
def login():
  return render_template('login.html')


@app.route('/check_login',methods=['POST'])
def check_route():
  username = request.form['username']
  password = request.form['password']
  if username=="admin" and password=="admin" : 
    return render_template('admin.html')
  return render_template('login.html',data="username or password faild")


@app.route('/influxDB', methods=['POST'])
def influxDB():
    try:
        token = request.form['api_token']
        url = request.form['url']
        org = request.form['org']
        measurement = request.form['measurement']
        bucket = request.form['brucket']
        csv = request.form['csv']
        nb_rows = request.form['nb_rows']
        
        infexdb = InfluxDB(token, org, url, bucket, measurement, csv,int(nb_rows))
        infexdb.getConnection()
        res = infexdb.import_CSV_DB()
        infexdb.GenerateModel()
        
        admin_url = url_for('admin', data='yes')  # Pass 'yes' when data is generated successfully
        return redirect(admin_url)
        
        
    except Exception as e:
        print(f"Error: {e}")
        return render_template('admin.html', error_message="yes")

@app.route('/statics')
def statics():
  return render_template('statistiques.html')

@app.route('/get_Chlorine_Time_data', methods = ['GET'])
def get_Chlorine_Time_data() :
  infexdb.getConnection()
  return jsonify(infexdb.get_Chlorine_Time_data())


if __name__ == "__main__" :
 app.run(debug=True)

'''
token = request.form['api_token']
  url = request.form['url']
  org = request.form['org']
  measurement = request.form['measurement']
  bucket = request.form['brucket']
  csv = request.form['csv']
  infexdb = InfluxDB(token,org,url,bucket,measurement,csv)
  infexdb.getConnection()
  res = infexdb.import_CSV_DB()
  infexdb.GenerateModel()
  return render_template('admin.html',data=res)
  #infexdb.displayAllDataInTerminal()
'''
