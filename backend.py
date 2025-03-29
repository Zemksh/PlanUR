from flask import Flask, url_for, render_template,request # type: ignore
app = Flask(__name__)
@app.route('/', methods = ['GET','POST'])
def home():
    print("HELLO WORLD")
    return render_template('index.html')

@app.route('/predict',methods = ['POST','GET'])
def predict():
    location = ""  # Default empty location
    if request.method == "POST":
        location = request.form.get("location")
    return render_template('index.html' , location = location, prediction_text='THIS IS HERE')


if __name__ == "__main__":
    app.run(debug=True)