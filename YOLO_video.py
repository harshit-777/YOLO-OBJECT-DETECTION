import cv2
import numpy as np
import time 


net = cv2.dnn.readNet("yolov3-tiny.weights","yolov3-tiny.cfg")
classes = []

with open("coco.names","r") as f:
	classes = [line.strip() for line in f.readlines()]

#print(classes)

layers_naems = net.getLayerNames()
output_layers = [layers_naems[i[0]-1] for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0,255,size=(len(classes),3))

cap = cv2.VideoCapture(0)
#adderss = "https://192.168.1.4:8080/video"
#cap.open(adderss)

time_now = time.time()
frame_id = 0 
font = cv2.FONT_HERSHEY_PLAIN

while True:
	_,frame = cap.read()
	frame_id +=1

	height,width,channels=frame.shape 

	#blob detect object
	blob = cv2.dnn.blobFromImage(frame,0.00392,(320,320),(0,0,0),True,crop=False)
	#we will get three differnt blob RGB color
	# for b in blob:
	# 	for n,imgg in enumerate(b):
	# 		cv2.imshow(str(n),imgg)


	net.setInput(blob)

	outs = net.forward(output_layers)

	#print(outs)		

	#showing info on pic of output

	boxes = []
	confidences = []
	class_ids=[]


	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]

			if confidence > 0.2:
				#object detected
				centre_x = int(detection[0] * width)
				centre_y = int(detection[1] * height)
				w = int(detection[2]*width)
				h = int(detection[3]*height)

				#cv2.circle(img,(centre_x,centre_y),10,(0,255,0),2)
				x = int(centre_x - w /2)
				y = int(centre_y -h/2)

				#cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

				boxes.append([x,y,w,h])
				confidences.append(float(confidence))
				class_ids.append(class_id)


	#print(len(boxes))		
	indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
	#print(indexes)	#print index we are taking 1,2,3,5 rest can remove
	font = cv2.FONT_HERSHEY_PLAIN
	object_dected = len(boxes)

	for i in range(len(boxes)):
		if i in indexes:
			x,y,w,h = boxes[i]
			label = str(classes[class_ids[i]])
			confidence = confidences[i]
			color = colors[class_ids[i]]
			print(label)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.putText(frame,label + "" + str(round(confidence,2)) ,(x,y+30),font,3,color,3)





	elapsed_time = time.time() - time_now 
	fps = frame_id / elapsed_time
	cv2.putText(frame,"FPS: " + str(round(fps,2)),(10,30),font,3,(0,0,0),2)
	cv2.imshow("Image",frame)
	key = cv2.waitKey(1)
	if key == 27:
		break

cap.release()

cv2.destroyAllWindows()