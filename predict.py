import cv2
import numpy as np

def predict(model, img):
    #random màu cho bounding box
    colors = np.random.uniform(0, 255, size=(len(model.classes), 3))

    #Xác định lớp output
    layer_names = model.net.getLayerNames()
    outputlayers = [layer_names[i[0] - 1] for i in model.net.getUnconnectedOutLayers()]

    height, width, _ = img.shape
    #tạo blob từ ảnh để đưa vào làm input
    blob = cv2.dnn.blobFromImage(
        img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    model.net.setInput(blob)
    outs = model.net.forward(outputlayers)
    IDclass = []
    confidences = []
    boxes = []

    #lặp qua mỗi lớp output
    for out in outs:
        #lặp qua từng đối tượng detect
        for detection in out:
            #lấy classid và confidence của đối tượng
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                IDclass.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences,0.5,0.3)

    title = "Objects: "
    font = cv2.FONT_HERSHEY_SIMPLEX
    if len(indexes) > 0:
		# lặp qua các vị trí indexes
        for i in indexes.flatten():
			# lấy toạ độ bounding box
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

			# vẽ bounding box
            color = [int(c) for c in colors[IDclass[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(model.classes[IDclass[i]], confidences[i])
            title += model.classes[IDclass[i]] + "  "
            cv2.putText(img, text, (x+3, y+15), font, 0.5, color, 2)

    return img, title
