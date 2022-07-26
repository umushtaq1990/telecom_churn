"""
A sample Hello World server.
"""
import os,sys
import pandas as pd
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
    try:
        df_res = Main_Pred(ID)
        if isinstance(df_res, pd.DataFrame):
            df_res = df_res.to_html(index=False,border=2,justify="center")
        html = '<center>' + \
                f'<h1> Telecom Churn Prediction for ID {ID} </h1><br><hr>' +\
                '<h2> Results </h2>' + df_res + '<be><hr>'+\
                '<h2> To get prediction score for given Customr ID append URL with : get_pred/ID </h2>' +\
                '</center>'
        return html
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    server_port = os.environ.get('PORT', '9090')
    app.run(debug=False, port=server_port, host='0.0.0.0')
