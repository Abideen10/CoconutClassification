import os
import numpy as np
import tensorflow as tf
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import random

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="/home/thedeener/MyTFLite/Model/soundclassifier_with_metadata.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Record audio from microphone
def record_audio(filename, duration=3, fs=44100):
    print("Recording...")
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    write(filename, fs, myrecording)  # Save as WAV file
    print("Recording finished")

# Make a prediction
def classify_audio(filename):
    data, samplerate = librosa.load(filename, sr=44100)
    expected_len = input_details[0]['shape'][1]
    
    if len(data) > expected_len:
        data = data[:expected_len]
    elif len(data) < expected_len:
        data = np.pad(data, (0, expected_len - len(data)), mode='constant')
    
    input_data = np.expand_dims(data, axis=0).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output_data), output_data

# Function to handle the button click
def on_record_button_click():
    record_button.pack_forget()
    
    filename = "recorded_audio.wav"
    record_audio(filename)
    label_index, probabilities = classify_audio(filename)
    
    with open("/home/thedeener/MyTFLite/Model/labels.txt", "r", encoding="utf-8") as f:
        labels = [label.strip() for label in f.readlines()]

    result = labels[label_index]

    # Set the formatted result based on the prediction
    if result == "มะพร้าวแก่":
        formatted_result = "มะพร้าวแก่"
        image_path = "/home/thedeener/MyTFLite/picture/no1.png"
    elif result == "มะพร้าวกลาง":
        formatted_result = "มะพร้าวกลาง"
        image_path = "/home/thedeener/MyTFLite/picture/no2.png"
    elif result == "มะพร้าวอ่อน":
        formatted_result = "มะพร้าวอ่อน"
        image_path = "/home/thedeener/MyTFLite/picture/no3.png"
    else:  # In case of "เสียงรบกวนในเบื้องหลัง"
        formatted_result = "ไม่มีการตรวจจับเสียงเคาะ"
        image_path = "/home/thedeener/MyTFLite/picture/noDetect.png"


    
    img = Image.open(image_path)
    img = img.resize((400, 200), Image.LANCZOS)  # Resize to fit the window size
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk, text='', width=450, height=250)
    image_label.image = img_tk
    
    root.after(3000, lambda: [
        image_label.config(image='', text=formatted_result, font=("Arial", 20), bg="lightgrey", width=35, height=4),
        record_button.pack(pady=5)
    ])

    print(f"Predicted: {result} with confidence {probabilities[0][label_index]}")

# Create the main window
root = tk.Tk()
root.title("Coconut Classification")
root.geometry("450x250")

title_label = tk.Label(root, text="The results of coconut", font=("Arial", 14))
title_label.pack(pady=5)

image_label = tk.Label(root, text="ผลลัพธ์", font=("Arial", 15), bg="lightgrey")
image_label.pack(pady=5)

record_button = tk.Button(root, text="คลิกเพื่อบันทึกเสียง", font=("Arial", 12), command=on_record_button_click, bg="lightgrey", width=30, height=2)
record_button.pack(side="bottom", pady=5)

root.mainloop()

