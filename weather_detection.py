import os
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
try:
    import win32com.client
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "weather_model.h5")
LABEL_FILE = os.path.join(BASE_DIR, "labels.txt")
IMAGE_SIZE = 128
selected_image_path = None
last_prediction = None
print("\n==============================================")
print("       SMART WEATHER CLASSIFICATION")
print("          CLOUDY AND SUNNY")
print("==============================================")
print("\nLoading weather model...")
if not os.path.exists(MODEL_FILE):
    print("\nERROR: Model file not found.")
    print("Expected location:", MODEL_FILE)
    input("\nPress Enter to close...")
    raise SystemExit
if not os.path.exists(LABEL_FILE):
    print("\nERROR: Label file not found.")
    print("Expected location:", LABEL_FILE)
    input("\nPress Enter to close...")
    raise SystemExit
try:
    model = tf.keras.models.load_model(MODEL_FILE)
    print("Model loaded successfully.")
except Exception as error:
    print("Model loading error:", error)
    input("\nPress Enter to close...")
    raise SystemExit
try:
    with open(LABEL_FILE, "r") as file:
        labels = [
            line.strip().lower()
            for line in file
            if line.strip()
        ]
except Exception as error:
    print("Label loading error:", error)
    input("\nPress Enter to close...")
    raise SystemExit
print("Labels:", labels)
print("Image Size:", IMAGE_SIZE)
if len(labels) != 2 or "cloudy" not in labels or "sunny" not in labels:
    root_error = tk.Tk()
    root_error.withdraw()
    messagebox.showerror(
        "Label Error",
        "The model must contain exactly these two classes:\n\n"
        "cloudy\nsunny\n\n"
        "Current labels:\n" + str(labels)
    )
    root_error.destroy()
    raise SystemExit
speech_engine = None
if SPEECH_AVAILABLE:
    try:
        speech_engine = win32com.client.Dispatch("SAPI.SpVoice")
        speech_engine.Rate = 0
        speech_engine.Volume = 100
        print("Windows Speech Engine: Ready")
    except Exception as error:
        print("Speech engine error:", error)
        speech_engine = None
def speak_result():
    global last_prediction
    if last_prediction is None:
        messagebox.showwarning(
            "Speech Predictor",
            "Please upload and classify an image first."
        )
        return
    weather = last_prediction["weather"]
    confidence = last_prediction["confidence"]
    if weather == "cloudy":
        speech_text = (
            "The weather prediction is Cloudy. "
            "The confidence is "
            + str(round(confidence))
            + " percent."
        )
    else:
        speech_text = (
            "The weather prediction is Sunny. "
            "The confidence is "
            + str(round(confidence))
            + " percent."
        )
    print("\n==============================================")
    print("              SPEECH PREDICTOR")
    print("==============================================")
    print("Speaking:", speech_text)
    if speech_engine is None:
        messagebox.showerror(
            "Speech Error",
            "Windows Speech Engine is not available.\n\n"
            "Please install pywin32 using:\n\n"
            "pip install pywin32"
        )
        return
    try:
        speech_engine.Speak("", 2)
        speech_engine.Speak(speech_text, 0)
    except Exception as error:
        print("Speech Error:", error)
        messagebox.showerror("Speech Error", str(error))
def select_image():
    global selected_image_path, last_prediction
    file_path = filedialog.askopenfilename(
        title="Select Weather Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
            ("JPEG Files", "*.jpg *.jpeg"),
            ("PNG Files", "*.png"),
            ("All Files", "*.*")
        ]
    )
    if not file_path:
        return
    try:
        image = Image.open(file_path).convert("RGB")
        image.thumbnail((550, 280))
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo, text="")
        image_label.image = photo
        selected_image_path = file_path
        last_prediction = None
        filename = os.path.basename(file_path)
        file_name_label.config(text="Selected: " + filename)
        cloudy_value.config(text="☁ Cloudy : --")
        sunny_value.config(text="☀ Sunny : --")
        prediction_value.config(
            text="READY TO CLASSIFY",
            fg="#00e6a8"
        )
        confidence_value.config(text="Confidence: --")
    except Exception as error:
        messagebox.showerror("Image Error", str(error))
def classify_image():
    global selected_image_path, last_prediction
    if selected_image_path is None:
        messagebox.showwarning(
            "No Image Selected",
            "Please select an image first."
        )
        return
    try:
        print("\n==============================================")
        print("             CLASSIFYING IMAGE")
        print("==============================================")
        print("Image:", selected_image_path)
        image = Image.open(selected_image_path).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        image_array = np.array(image, dtype=np.float32)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        print("Input shape:", image_array.shape)
        prediction = model.predict(image_array, verbose=0)
        probabilities = prediction[0]
        print("Raw prediction:", probabilities)
        predicted_index = int(np.argmax(probabilities))
        predicted_weather = labels[predicted_index]
        confidence = float(probabilities[predicted_index]) * 100
        cloudy_index = labels.index("cloudy")
        cloudy_probability = float(probabilities[cloudy_index]) * 100
        sunny_index = labels.index("sunny")
        sunny_probability = float(probabilities[sunny_index]) * 100
        last_prediction = {
            "weather": predicted_weather,
            "confidence": confidence,
            "cloudy": cloudy_probability,
            "sunny": sunny_probability
        }
        print("Cloudy:", round(cloudy_probability, 2), "%")
        print("Sunny:", round(sunny_probability, 2), "%")
        print("Prediction:", predicted_weather)
        print("Confidence:", round(confidence, 2), "%")
        cloudy_value.config(
            text=f"☁ Cloudy : {cloudy_probability:.2f}%"
        )
        sunny_value.config(
            text=f"☀ Sunny : {sunny_probability:.2f}%"
        )
        if predicted_weather == "cloudy":
            prediction_value.config(
                text="☁ CLOUDY",
                fg="#60a5fa"
            )
        else:
            prediction_value.config(
                text="☀ SUNNY",
                fg="#facc15"
            )
            confidence_value.config(
            text=f"Confidence: {confidence:.2f}%"
        )
        messagebox.showinfo(
            "Weather Classification Result",
            "🌤 WEATHER CLASSIFICATION\n\n"
            + f"☁ Cloudy : {cloudy_probability:.2f}%\n\n"
            + f"☀ Sunny : {sunny_probability:.2f}%\n\n"
            + "----------------------------\n\n"
            + "Prediction : "
            + predicted_weather.upper()
            + "\n\n"
            + f"Confidence : {confidence:.2f}%"
        )
    except Exception as error:
        print("\nClassification Error:", error)
        messagebox.showerror("Classification Error", str(error))
def reset_all():
    global selected_image_path, last_prediction

    selected_image_path = None
    last_prediction = None
    image_label.config(
        image="",
        text="📷\n\nSELECT A WEATHER IMAGE"
    )
    image_label.image = None
    file_name_label.config(text="No image selected")
    cloudy_value.config(text="☁ Cloudy : --")
    sunny_value.config(text="☀ Sunny : --")
    prediction_value.config(text="READY", fg="#00e6a8")
    confidence_value.config(text="Confidence: --")
def exit_program():
    result = messagebox.askyesno(
        "Exit",
        "Are you sure you want to exit?"
    )
    if result:
        root.destroy()
root = tk.Tk()
root.title("Smart Weather Classification")
root.geometry("780x750")
root.resizable(False, False)
root.configure(bg="#111827")
header = tk.Frame(
    root,
    bg="#7c3aed",
    height=90
)
header.pack(fill="x")
header.pack_propagate(False)
title_label = tk.Label(
    header,
    text="🌤 SMART WEATHER CLASSIFICATION",
    font=("Arial", 23, "bold"),
    fg="white",
    bg="#7c3aed"
)
title_label.pack(pady=(14, 2))
subtitle_label = tk.Label(
    header,
    text="Cloudy and Sunny Image Detection using CNN",
    font=("Arial", 11),
    fg="#ede9fe",
    bg="#7c3aed"
)
subtitle_label.pack()
image_frame = tk.Frame(
    root,
    bg="#1f2937",
    highlightbackground="#8b5cf6",
    highlightthickness=2,
    width=640,
    height=310
)
image_frame.pack(pady=12)
image_frame.pack_propagate(False)
image_label = tk.Label(
    image_frame,
    text="📷\n\nSELECT A WEATHER IMAGE",
    font=("Arial", 16, "bold"),
    fg="#c4b5fd",
    bg="#1f2937"
)
image_label.pack(expand=True)
file_name_label = tk.Label(
    root,
    text="No image selected",
    font=("Arial", 10),
    fg="#9ca3af",
    bg="#111827"
)
file_name_label.pack(pady=(0, 4))
result_frame = tk.Frame(root, bg="#111827")
result_frame.pack(pady=2)
cloudy_value = tk.Label(
    result_frame,
    text="☁ Cloudy : --",
    font=("Arial", 14, "bold"),
    fg="#60a5fa",
    bg="#111827"
)
cloudy_value.grid(row=0, column=0, padx=25)
sunny_value = tk.Label(
    result_frame,
    text="☀ Sunny : --",
    font=("Arial", 14, "bold"),
    fg="#facc15",
    bg="#111827"
)
sunny_value.grid(row=0, column=1, padx=25)
prediction_value = tk.Label(
    root,
    text="READY",
    font=("Arial", 22, "bold"),
    fg="#00e6a8",
    bg="#111827"
)
prediction_value.pack(pady=(7, 1))
confidence_value = tk.Label(
    root,
    text="Confidence: --",
    font=("Arial", 11, "bold"),
    fg="#d1d5db",
    bg="#111827"
)
confidence_value.pack(pady=2)
button_frame = tk.Frame(root, bg="#111827")
button_frame.pack(pady=8)
upload_button = tk.Button(
    button_frame,
    text="📂 UPLOAD IMAGE",
    command=select_image,
    font=("Arial", 11, "bold"),
    bg="#8b5cf6",
    fg="white",
    activebackground="#7c3aed",
    activeforeground="white",
    width=17,
    height=2,
    relief="flat",
    cursor="hand2"
)
upload_button.grid(row=0, column=0, padx=5)
classify_button = tk.Button(
    button_frame,
    text="🔍 CLASSIFY IMAGE",
    command=classify_image,
    font=("Arial", 11, "bold"),
    bg="#10b981",
    fg="white",
    activebackground="#059669",
    activeforeground="white",
    width=17,
    height=2,
    relief="flat",
    cursor="hand2"
)
classify_button.grid(row=0, column=1, padx=5)
speech_button = tk.Button(
    button_frame,
    text="🔊 SPEAK RESULT",
    command=speak_result,
    font=("Arial", 11, "bold"),
    bg="#f59e0b",
    fg="white",
    activebackground="#d97706",
    activeforeground="white",
    width=17,
    height=2,
    relief="flat",
    cursor="hand2"
)
speech_button.grid(row=0, column=2, padx=5)
button_frame2 = tk.Frame(root, bg="#111827")
button_frame2.pack(pady=3)
reset_button = tk.Button(
    button_frame2,
    text="🔄 RESET",
    command=reset_all,
    font=("Arial", 11, "bold"),
    bg="#64748b",
    fg="white",
    activebackground="#475569",
    activeforeground="white",
    width=18,
    height=2,
    relief="flat",
    cursor="hand2"
)
reset_button.grid(row=0, column=0, padx=8)
exit_button = tk.Button(
    button_frame2,
    text="❌ EXIT",
    command=exit_program,
    font=("Arial", 11, "bold"),
    bg="#ef4444",
    fg="white",
    activebackground="#dc2626",
    activeforeground="white",
    width=18,
    height=2,
    relief="flat",
    cursor="hand2"
)
exit_button.grid(row=0, column=1, padx=8)
footer_label = tk.Label(
    root,
    text="Artificial Intelligence • CNN • Weather Image Classification",
    font=("Arial", 9),
    fg="#6b7280",
    bg="#111827"
)
footer_label.pack(pady=8)
root.mainloop()