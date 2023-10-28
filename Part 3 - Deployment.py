from tkinter import filedialog
from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image, ImageOps # Install pillow instead of PIL
from keras.models import load_model # TensorFlow is required for Keras to work
import numpy as np

#load the trained model
model = load_model('keras_model.h5')
# Load the labels
label_path='labels.txt'
class_names = open(label_path, "r").readlines()

test_image_path = 'test-image.jpg'
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

#initialize GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Male or Female celebrity image classification')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
output_image = Label(top)

def classify(test_image_path):
    global label_packed
    disp_string= ''
    image = Image.open(test_image_path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display prediction and confidence score
    disp_string+= "\nClass:"+ str(class_name[2:])
    disp_string+= "\nConfidence Score:"+ str(confidence_score)

    #label.configure(foreground='#011638', text=class_name)
    label.configure(foreground='#011638', text=disp_string)

def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        output_image.configure(image=im)
        output_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
output_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Male or Female Celebrity Image Classifier",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()

