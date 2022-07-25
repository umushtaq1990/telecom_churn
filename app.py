"""
A sample Hello World server.
"""
import os
from flask import Flask, render_template
from get_pred import Main_Pred

app = Flask(__name__)


@app.route('/')
def default_page():
    message = '<center><h1> Telecom Churn Prediction Model </h1>' +\
              '<h2> To get prediction score for given Customr ID append URL with : get_pred/ID </h2>' +\
              '</center>'
    return message

@app.route('/get_pred/<ID>')
def Get_Telecom_Churn_Pred(ID):
    df_res = Main_Pred(ID)
    message = f'<center><h1> Telecom Churn Prediction for ID {ID} </h1>' +\
              '<h2> {df_res.to_html()} </h2>' +\
              '</center>'
    return message

if __name__ == '__main__':
    server_port = os.environ.get('PORT', '8082')
    app.run(debug=False, port=server_port, host='0.0.0.0')
