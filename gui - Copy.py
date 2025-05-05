import tkinter as tk
from tkinter import filedialog
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from PIL import Image, ImageTk

MODEL_PATH = 'pythonProject/fish_classifier.h5'
IMG_SIZE = (128, 128)
CLASSES = [
    'Black Sea Sprat', 'Gilt-Head Bream', 'Hourse Mackerel',
    'Red Mullet', 'Red Sea Bream', 'Sea Bass', 'Shrimp',
    'Striped Red Mullet', 'Trout'
]


def preprocess_for_prediction(img_bgr):
    """
    Preprocess only for model input, not display.
    """
    img_resized = cv2.resize(img_bgr, IMG_SIZE)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    _, thresh = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return img_resized, contours


def browse_and_predict():
    filepath = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[('Image Files', '*.png;*.jpg;*.jpeg;*.bmp')]
    )
    if not filepath:
        return

    # Load model
    model = load_model(MODEL_PATH)

    # Load full-size image (for display)
    img_bgr_full = cv2.imread(filepath)
    img_display = img_bgr_full.copy()

    # Preprocess for prediction (use resized version)
    img_resized, contours = preprocess_for_prediction(img_bgr_full)

    # Predict on resized image
    img_input = img_resized.astype('float32') / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    preds = model.predict(img_input)
    idx = np.argmax(preds, axis=1)[0]
    cls = CLASSES[idx]
    prob = preds[0][idx]

    # Calculate scaling factors
    full_h, full_w = img_bgr_full.shape[:2]
    scale_x = full_w / IMG_SIZE[0]
    scale_y = full_h / IMG_SIZE[1]

    # Draw bounding box on original large image
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Scale bounding box back to original size
        x_full = int(x * scale_x)
        y_full = int(y * scale_y)
        w_full = int(w * scale_x)
        h_full = int(h * scale_y)

        cv2.rectangle(img_display, (x_full, y_full), (x_full + w_full, y_full + h_full), (0, 0, 255), 4)

    # Draw label on top
    cv2.putText(
        img_display,
        f'{cls} ({prob:.2f})',
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 255),
        3
    )

    # Resize display image if too large
    max_display_size = 600
    height, width = img_display.shape[:2]
    if max(height, width) > max_display_size:
        scale = max_display_size / max(height, width)
        img_display = cv2.resize(img_display, (int(width * scale), int(height * scale)))

    # Convert BGR to RGB for Tkinter
    img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tk_img = ImageTk.PhotoImage(pil_img)

    # Update GUI
    image_label.config(image=tk_img)
    image_label.image = tk_img  # keep reference!
    result_label.config(text=f'Predicted: {cls} (Confidence: {prob:.2f})')

    print(f'Predicted: {cls}, Confidence: {prob:.2f}')


if __name__ == '__main__':
    root = tk.Tk()
    root.title('Fish Classifier')
    root.geometry('800x800')

    browse_button = tk.Button(
        root,
        text='Select Image',
        command=browse_and_predict,
        width=20,
        pady=5
    )
    browse_button.pack(pady=10)

    image_label = tk.Label(root)
    image_label.pack(pady=10)

    result_label = tk.Label(root, text='No image selected.')
    result_label.pack(pady=10)

    root.mainloop()


