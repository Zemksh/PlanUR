from flask import Flask, render_template, request, Response, url_for
from ML.stage1_1 import main 

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    print("HELLO WORLD")
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    global latitude 
    latitude= request.form.get("latitude", "").strip()
    global longitude 
    longitude= request.form.get("longitude", "").strip()

    if not latitude or not longitude:
        return render_template('index.html', noinput="Enter valid latitude and longitude")

    return render_template('prediction.html', latitude=latitude, longitude=longitude)

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
    img = main(28.6139, 77.2090)  # Example coordinates for Delhi
    if img is None:
        return "❌ Error generating image", 500
    return Response(img.getvalue(), mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)
