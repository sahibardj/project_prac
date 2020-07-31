import cv2
import numpy as np
#initialize camera
cap=cv2.VideoCapture(0)
#face  detection
face_cascade= cv2.CascadeClassifier("hfd.xml")
skip=0
face_data=[]
dataset_path= './data/'
file_name=input("enter name")

while True:
	ret,frame=cap.read()
	if ret== False:
		continue
	faces= face_cascade.detectMultiScale(frame,1.3,5)
	faces=sorted(faces,key=lambda f: f[2]*f[3])
	#pick the largest face(because it is the larget face according to area)
	for face in faces[-1:]:
		x,y,w,h=face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
		#Extract (Crop out the required face): region of intrest
		offset=10
		face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
		fs=cv2.resize(face_section,(100,100))
		skip+=1
		if skip%10==0:
			face_data.append(fs)
			print(len(face_data))
	cv2.imshow("frame",frame)
	key_pressed=cv2.waitKey(1)& 0xFF
	if key_pressed== ord('q'):
		break
#convert our face list array into a numpy array
face_data=np.asarray(face_data)
face_data= face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)
#save this data in a file
np.save(dataset_path+file_name+'.npy', face_data)
print("data sucessfully saved at ",dataset_path+file_name)
cap.release()
cv2.destroyAllWindow() 
