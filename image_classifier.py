import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import os
import cv2

# Ustawienie ścieżki do modelu
model_path = os.path.join('models', 'screws.h5')

# Wczytaj nauczony model
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading the model: {e}")
    exit(1)

# Funkcja do klasyfikacji obrazu
def classify_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resize = tf.image.resize(img_rgb, (256, 256))
    yhat = model.predict(np.expand_dims(resize / 255.0, 0))
    if yhat > 0.5:
        return "Wskazany obraz został sklasyfikowany jako część prawidłowa"
    else:
        return "Wskazany obraz został sklasyfikowany jako część uszkodzona"

# Funkcja do wczytania obrazu
def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            img = Image.open(file_path)
            img.thumbnail((400, 400))  # Zwiększenie maksymalnego rozmiaru wyświetlanego obrazu
            img = ImageTk.PhotoImage(img)
            panel.configure(image=img)
            panel.image = img
            
            # Klasyfikacja obrazu
            result_message = classify_image(file_path)
            result_label.configure(text=result_message)
        except Exception as e:
            messagebox.showerror("Error", f"Unable to open image file.\nError: {e}")

customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("blue")

# Tworzenie głównego okna
app = customtkinter.CTk()
app.title("Image Classifier")
app.geometry("800x600")

# Panel do wyświetlania obrazu
panel = customtkinter.CTkLabel(app, text="")
panel.pack(padx=10, pady=10, expand=True)

# Etykieta do wyświetlania wyniku klasyfikacji
result_label = customtkinter.CTkLabel(app, text="")
result_label.pack(pady=10)

# Tworzenie przycisku do wczytywania obrazu i wyśrodkowanie go
btn = customtkinter.CTkButton(app, corner_radius=0, text="Load Image", command=load_image)
btn.pack(pady=20)  # Dodatkowe miejsce nad i pod przyciskiem

# Uruchomienie aplikacji
app.mainloop()