import cv2
import numpy as np
import time

yolo = cv2.dnn.readNet('./yolo v3/yolov3.weights', './yolo v3/yolov3.cfg')

with open('./yolo v3/coco.names') as f:
    classes = f.read().splitlines()


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()

    start = time.time()

    blob = cv2.dnn.blobFromImage(image, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)

    yolo.setInput(blob)

    output_layers_name = yolo.getUnconnectedOutLayersNames()
    layeroutput = yolo.forward(output_layers_name)

    boxes = []
    confidences = []
    class_ids = []

    w, h = image.shape[:2]

    for output in layeroutput:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]

            if confidence > 0.5:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]

                boxes.append(box)
                confidences.append(float(confidence))
                class_ids.append(class_id)

    if len(boxes) != 0:
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        colors = np.random.uniform(0, 255, size=(len(boxes), 3))

        for i in indexes.flatten():
            x, y, w, h = boxes[i]

            label = str(classes[class_ids[i]])
            confi = str(round(confidences[i], 2))
            color = colors[i]
            print(label, confi)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 1)
            cv2.putText(image, label + ' ' + confi, (x, y+20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)

    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime

    cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    cv2.imshow('Object Detection', image)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
