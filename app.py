from flask import Flask, render_template, request,send_file
import sqlite3
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATABASE'] = 'database.db'
app.config['SECRET_KEY'] = 'boost is the sectret of my energy'

def create_table():
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS data
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      text TEXT NOT NULL,
                      image BLOB NOT NULL)''')
    conn.commit()
    conn.close()

# # Save the uploaded image as a BLOB
def save_image(image):
    return image.read()

# Route to handle the form submission
@app.route("/", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        text = request.form['text']
        image = request.files['image']
        #image_data = None

        if text and image:
            create_table()
            image_data = save_image(image)

            conn = sqlite3.connect(app.config['DATABASE'])
            cursor = conn.cursor()
            cursor.execute("INSERT INTO data (text, image) VALUES (?, ?)", (text, image_data))
            conn.commit()
            conn.close()


            
            model = YOLO("yolov8n.pt")
            #img = mpimg.imread(image)
            img = Image.open(image)
            img = img.convert('RGB')

            # Convert the PIL Image to a numpy array if required
            img_array = np.array(img)
            results=model.predict(img_array)
            #results=model.predict(file_path)

            #imgplot = plt.imshow(img)
            #plt.show()
            no_objts=len(results[0].boxes)
            objs=[]
            for i in range(no_objts):
                objs.append(results[0].names[results[0].boxes[i].cls[0].item()])
            from collections import Counter
            counter_result = Counter(objs)

            result_string = str(counter_result)
            result_content = result_string[result_string.index('{')+1:result_string.index('}')]

            return f"<h1>{result_content}</h1>"

    return render_template('index.html')
    

if __name__ == '__main__':
    app.run(debug=True)
