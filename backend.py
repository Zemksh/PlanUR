from flask import Flask, render_template, request, Response, url_for
from ML.stage1_1 import produce_image 
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
    if location =="":
        return render_template('index.html' , noinput = "enter a valid input")
    else:
        return render_template('prediction.html' , location = location)
  

@app.route('/final', methods=['POST'])
def final():
    answer = request.form.get("answer")
    if answer == "greencover":
        return render_template('prediction.html', answer=answer, Textbe="Green cover map", image_url=url_for('plot'))
    elif answer == "renewableenergy":
        return render_template('prediction.html', answer=answer, Textbe="Renewable energy map", image_url=url_for('plot'))
    elif answer == "balanced":
        return render_template('prediction.html', answer=answer, Textbe="Balanced map", image_url=url_for('plot'))
    else:
        return render_template('prediction.html', error="Please select an option")
@app.route('/plot.png')
def plot():
    img = produce_image()  # ✅ Call the function correctly
    return Response(img.getvalue(), mimetype='image/png')
if __name__ == "__main__":
    app.run(debug=True)