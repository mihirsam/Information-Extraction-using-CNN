from flask import Flask, render_template, request
from PredictionClass_WorldWar2 import Predict

app = Flask(__name__, template_folder='template')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/',  methods = ['POST', 'GET'])
def home():
    summ = "No File"
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)

        summ = Predict(f.filename)
    return render_template('index.html', summary=summ)


if __name__ == '__main__':
    app.run(debug=True)