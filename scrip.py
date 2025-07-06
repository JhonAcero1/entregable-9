import cv2
import numpy as np
 
# Cargar nombres de clases
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
 
# Objetos de interés
objetos_interes = ['person', 'cell phone', 'mouse']
 
# Cargar red YOLOv4
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
 
# Colores
colors = np.random.uniform(0, 255, size=(len(classes), 3))
 
# Iniciar cámara
cap = cv2.VideoCapture(0)
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    height, width, _ = frame.shape
 
    # Preparar blob
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
 
    class_ids = []
    confidences = []
    boxes = []
 
    # Detección
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            label = classes[class_id]
 
            if confidence > 0.5 and label in objetos_interes:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
 
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
 
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
 
    # Contador de objetos
    contador = {'person': 0, 'cell phone': 0, 'mouse': 0}
 
    if len(indexes) > 0:
        for i in indexes:
            i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
 
            if label in objetos_interes:
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                contador[label] += 1
 
    # Mostrar conteo en tiempo real
    texto = f"person: {contador['person']} | cell phone: {contador['cell phone']} | mouse: {contador['mouse']}"
    cv2.putText(frame, texto, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
 
    # ESTO GUARDARA LA IMAGEN
    if contador['person'] >= 2 and contador['cell phone'] == 1 and contador['mouse'] == 0:
        cv2.imwrite("mobilenet_filtro_resultado.jpg", frame)
 
    # Mostrar ventana
    cv2.imshow("Detección precisa", frame)
 
    # Salir con tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# Cerrar recursos
cap.release()
cv2.destroyAllWindows()