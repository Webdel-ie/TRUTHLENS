from flask import Flask, request, render_template
from u_model_interface import predict

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    text = request.form['news_text']
    result = predict(text)
    return render_template('index.html', prediction=result, news_text=text)

if __name__ == "__main__":
    app.run(debug=True)
