from flask import Flask,render_template,url_for,request,redirect
import pandas as pd
import joblib
app = Flask(__name__)

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


if __name__ == "__main__" :
 app.run(debug=True)
