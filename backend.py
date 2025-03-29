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
    return render_template('prediction.html' , location = location, prediction_text='THIS IS HERE')

@app.route('/final',methods =['GET','POST'])
def final():
    answer = ""
    if request.method =="POST":
        answer = request.form.get("answer")
    if answer == "greencover":
        return  render_template('prediction.html',answer = answer,Textbe="Green cover map")
    elif answer == "renewableenergy":
        return  render_template('prediction.html',answer = answer,Textbe="Renewable energy map ")
    elif answer == "balanced":
        return  render_template('prediction.html',answer = answer, Textbe="balanced map be")
    else:
        return render_template('prediction.html',answer = answer, error="please select an option")
if __name__ == "__main__":
    app.run(debug=True)