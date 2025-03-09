from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import torch
from PIL import Image
import numpy as np
import cv2
import io
import pathlib
from base64 import b64encode
import json

# Solución temporal para Windows (si es necesario)
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Modelo Pydantic para validar los parámetros de configuración
class DrawingConfig(BaseModel):
    line_thickness: Optional[float] = 1
    alpha: Optional[float] = 0.3
    font_scale: Optional[float] = 0.5
    font_thickness: Optional[float] = 1.0

# Configuración de la aplicación FastAPI
app = FastAPI(title="Object Detection API", description="API para detección de objetos con YOLOv5")

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Colores para las clases detectadas
COLORS = {
    "RBC": (0, 255, 0),   # Verde
    "WBC": (255, 0, 0),   # Rojo
    "Platelet": (0, 0, 255)  # Azul
}

# Cargar el modelo al iniciar la API
@app.on_event("startup")
async def load_model():
    try:
        # Aplicar fix para Windows
        pathlib.PosixPath = pathlib.WindowsPath
        
        # Cargar modelo YOLOv5
        app.state.model = torch.hub.load(
            "ultralytics/yolov5", 
            "custom", 
            path="runs/train/exp/weights/best.pt",
            force_reload=True
        )
        
        # Restaurar PosixPath
        pathlib.PosixPath = temp
    except Exception as e:
        raise RuntimeError(f"Error cargando el modelo: {str(e)}")

# Función para procesar la imagen
async def process_image(image: Image.Image, config: DrawingConfig = None):
    """Procesa la imagen y realiza las detecciones"""
    try:
        # Convertir a numpy array
        image_np = np.array(image)
        
        # Realizar detección
        results = app.state.model(image_np)
        detections = results.xyxy[0]
        
        # Crear imagen con bounding boxes
        image_with_boxes = image_np.copy()
        
        # Usar los parámetros de configuración o los valores por defecto
        line_thickness = config.line_thickness if config else 1
        alpha = config.alpha if config else 0.3
        font_scale = config.font_scale if config else 0.5
        font_thickness = config.font_thickness if config else 1
        
        # Dibujar detecciones
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            class_name = app.state.model.names[int(cls)]
            color = COLORS.get(class_name, (0, 255, 0))
            
            # Dibujar cuadro con transparencia
            overlay = image_with_boxes.copy()
            cv2.rectangle(
                overlay,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                int(line_thickness),  # Convertir a entero
            )
            cv2.addWeighted(overlay, alpha, image_with_boxes, 1 - alpha, 0, image_with_boxes)
            
            # Dibujar etiqueta
            label = f"{class_name} {conf:.2f}"
            cv2.putText(
                image_with_boxes,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                int(font_thickness),  # Convertir a entero
            )
        
        # Convertir la imagen de BGR a RGB
        #image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        
        return image_with_boxes, detections
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")

# Endpoint para detectar objetos en una imagen
@app.post("/detect/", summary="Detectar objetos en una imagen")
async def detect_objects(
    file: UploadFile = File(..., description="Imagen para analizar"),
    config: str = Form(None)  # Recibir el campo "config" como cadena JSON
):
    """Endpoint para detección de objetos en imágenes usando YOLOv5"""
    
    # Validar tipo de archivo
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Formato de archivo no soportado")
    
    try:
        # Leer y procesar imagen
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Parsear la configuración si se proporciona
        drawing_config = None
        if config:
            config_dict = json.loads(config)
            drawing_config = DrawingConfig(**config_dict)
        
        # Procesar imagen
        processed_image, detections = await process_image(image, drawing_config)
        
        # Convertir imagen a base64
        pil_image = Image.fromarray(processed_image)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_base64 = b64encode(buffered.getvalue()).decode("utf-8")
        
        # Preparar respuesta
        detections_list = [
            {
                "class": app.state.model.names[int(det[5])],
                "confidence": float(det[4]),
                "bbox": {
                    "x1": float(det[0]),
                    "y1": float(det[1]),
                    "x2": float(det[2]),
                    "y2": float(det[3])
                }
            }
            for det in detections
        ]
        
        return JSONResponse(content={
            "detections": detections_list,
            "count": len(detections),
            "image_base64": img_base64,
            "format": "image/jpeg",
            "config": config
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Ejecutar la aplicación
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)