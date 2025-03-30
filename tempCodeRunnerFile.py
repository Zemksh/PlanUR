@app.route('/predict',methods = ['POST','GET'])
def predict():
    location = ""  # Default empty location
    if request.method == "POST":
        location = request.form.get("location")
    if location =="":
        return render_template('index.html' , noinput = "enter a valid input")
    else:
        return render_template('prediction.html' , location = location)