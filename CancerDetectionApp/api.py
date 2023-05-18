from flask import Flask
from flask import render_template
from flask import request
import os
import tensorflow as tf
from numpy import asarray
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = "C:\Cancer Detection\CancerDetectionApp\static"

@app.route("/",methods=["GET","POST"])
def upload_predict():
    if request.method == "POST":
        global model
        model = tf.keras.models.load_model('C:\Cancer Detection\CancerDetectionApp\my_model.h5')
        print(" * Model loaded!")
        image_file = request.files["image"]
        if image_file :
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            prediction=0
            image_file.save(image_location)
            print(image_location)
            image = Image.open(image_location)
            data = asarray(image)
            data_reshaped = data.reshape(1,96,96,3,)
            prediction = model.predict(data_reshaped,steps=5,verbose=1)
            print(prediction)
            prediction2 = "Some Error Occurred"

            if(prediction[0][0]>0.5):
                prediction2 = "The patient is diagnosed with Cancer"
            else:
                prediction2 = "The patient is not diagnosed with cancer."
            return render_template("index.html", prediction=prediction2)
    return render_template("index.html",prediction="Select an Image" )

if __name__ == "__main__":
    app.run(port=3500,debug=True)