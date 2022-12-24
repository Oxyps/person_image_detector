import cv2
import imutils
import numpy as np


NMS_THRESHOLD=0.3
MIN_CONFIDENCE=0.2
LABELS = open('./objetos.names').read().strip().split('\n')
INDEX_PERSON_LABEL = LABELS.index('person')


image = cv2.imread('./inputs/pedestre_noite.jpg')
image = imutils.resize(image, width=700)

model = cv2.dnn.readNet('./modelo/yolov4-tiny.cfg', './modelo/yolov4-tiny.weights')

layer_name = model.getLayerNames()
layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]

(H, W) = image.shape[:2]
results = []

blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), swapRB=True, crop=False)
model.setInput(blob)
layer_outputs = model.forward(layer_name)

boxes = []
centroids = []
confidences = []

for output in layer_outputs:
	for detection in output:
		scores = detection[5:]
		class_id = np.argmax(scores)
		confidence = scores[class_id]

		# Atribui a external_object a string da posição classID em objetos.names
		external_object = LABELS[class_id]

		# Se esse objeto for diferente de uma pessoa faz print de qual objeto é.
		if external_object != LABELS[INDEX_PERSON_LABEL]: 
			print(external_object) 

		if class_id == INDEX_PERSON_LABEL and confidence > MIN_CONFIDENCE:
			box = detection[:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype('int')

			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			boxes.append([x, y, int(width), int(height)])
			centroids.append((centerX, centerY))
			confidences.append(float(confidence))
			print(confidence)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)

	# loop over the indexes we are keeping
	for i in idzs.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])
		# update our results list to consist of the person
		# prediction probability, bounding box coordinates,
		# and the centroid
		result = (confidences[i], (x, y, x + w, y + h), centroids[i])
		results.append(result)

font = cv2.FONT_HERSHEY_COMPLEX

for result in results:
	colors = np.random.uniform(0, 255, size=(80, 3))
	color = colors[i]

	cv2.rectangle(image, (result[1][0],result[1][1]), (result[1][2],result[1][3]), color, 2)
	cv2.putText(image, '', (result[1][0], result[1][1] - 20), font, 1, color, 2)

cv2.imshow('Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
