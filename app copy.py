import streamlit as st
import torch
import pathlib
from PIL import Image
import numpy as np
import cv2

# Solución para Windows: Reemplazar PosixPath con WindowsPath
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Título de la aplicación
st.title("Detector de Objetos con YOLOv5")

# Cargar el modelo YOLOv5
@st.cache_resource
def load_model():
    model = torch.hub.load("ultralytics/yolov5", "custom", path="runs/train/exp/weights/best.pt")
    return model

model = load_model()

# Restaurar PosixPath (opcional)
pathlib.PosixPath = temp

# Subir una imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Leer la imagen
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen subida', use_column_width=True)

    # Convertir la imagen a formato compatible con YOLOv5
    image_np = np.array(image)  # Convertir a numpy array

    # Ejecutar el modelo
    results = model(image_np)

    # Obtener las detecciones
    detections = results.xyxy[0]

    # Crear una copia de la imagen para dibujar los cuadros
    image_with_boxes = image_np.copy()

    # Configuración personalizada
    line_thickness = 1  # Grosor de los cuadros (más delgado)
    alpha = 1  # Transparencia (0 = completamente transparente, 1 = completamente opaco)
    font_scale = 0.5  # Tamaño de la letra de las etiquetas
    font_thickness = 1  # Grosor de la letra

    # Colores personalizados para cada clase
    colors = {
        "RBC": (0, 255, 0),  # Verde para glóbulos rojos
        "WBC": (255, 0, 0),  # Rojo para glóbulos blancos
        "Platelet": (0, 0, 255),  # Azul para plaquetas
        # Agrega más clases y colores según sea necesario
    }

    # Dibujar los cuadros con transparencia y etiquetas
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        class_name = model.names[int(cls)]
        color = colors.get(class_name, (0, 255, 0))  # Usar verde como color predeterminado si la clase no está en el diccionario
        
        # Dibujar el cuadro con transparencia
        overlay = image_with_boxes.copy()
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, line_thickness)
        cv2.addWeighted(overlay, alpha, image_with_boxes, 1 - alpha, 0, image_with_boxes)

        # Agregar etiquetas con el nombre de la clase y la confianza
        label = f"{class_name} {conf:.2f}"
        cv2.putText(
            image_with_boxes,  # Imagen sobre la que se dibuja
            label,  # Texto de la etiqueta
            (int(x1), int(y1) - 10),  # Posición de la etiqueta (arriba del cuadro)
            cv2.FONT_HERSHEY_SIMPLEX,  # Tipo de fuente
            font_scale,  # Tamaño de la letra
            color,  # Color de la letra
            font_thickness,  # Grosor de la letra
        )

    # Mostrar las detecciones
    st.subheader("Resultados de la detección")
    st.image(image_with_boxes, caption='Detecciones', use_column_width=True)

    # Mostrar estadísticas
    st.write(f"**Número de detecciones:** {len(detections)}")
    for detection in detections:
        class_id = int(detection[5])
        class_name = model.names[class_id]
        confidence = float(detection[4])
        st.write(f"**Clase:** {class_name}, **Confianza:** {confidence:.2f}")