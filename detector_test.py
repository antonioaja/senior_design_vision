from ultralytics import YOLO
import cv2
import time

# Start webcam 
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(3,480)


# Load our custom bottle detection deep learning model
model = YOLO("runs/detect/train4/weights/best.pt")
 
classNames = ["can", "plastic bottle", "nothing", "nothing"]

frames = 0

while True:
    if frames == 0:
        start = time.time()

    success, img= cap.read()

    results = model.predict(img, stream=True)

    frames = frames + 1

    if time.time() - start >= 1.0:
        print("%d FPS" % frames)
        frames = 0
#    print('1')

#    if results.isEmpty() != False:
#    if not results:
#        for r in results:
#            boxes = r.boxes
#            print('2')

#        for box in boxes:
#            # bounding box
#            x1, y1, x2, y2 = box.xyxy[0]
#            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
#            print('3')
#            # put box in cam
#            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
#
#            # confidence
#            confidence = math.ceil((box.conf[0]*100))/100
#            print("Confidence -->",confidence)
#
#            # class name
#            cls = int(box.cls[0])
#            print("Class name -->", classNames[cls])
#
#            # object details
#            org = [x1, y1]
#            font = cv2.FONT_HERSHEY_SIMPLEX
#            fontScale = 1
#            color = (255, 0, 0)
#            thickness = 2
#            text = classNames[cls] + "-> Confidence: " + str(confidence)
#
#            cv2.putText(img, text, org, font, fontScale, color, thickness)

    #success, jpg = cv2.imencode('.jpg', results[0].plot())
    #cv2.imshow('Bottle Detector', jpg)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
