#read live video stream
#extract face oout of trining data
import numpy as np
import cv2
import os
 # KNN CODE
def distance(v1,v2):
 	#eucledian
 	return np.sqrt(((v1-v2)**2).sum())
def knn(train,test,k=5):
 	dist=[]
 	for i in range(train.shape[0]):
 		#get the vector and lael
 		ix= train[i, :-1]
 		iy= train[i, -1]
 		#compute distnce from test point
 		d= distance(test,ix)
 		dist.append([d,iy])
 	# to be sorted on base of d and get fisrt k (odd to avoid dispute)
 	dk= sorted(dist, key= lambda x: x[0])[:k]
 	#retrive only the labels
 	labels= np.array(dk)[:,-1]
 	#get frequesncy of each label
 	output=np.unique(labels,return_counts=True)
 	#find max Frequencyand its label
 	index=np.argmax(output[1])
 	return output[0][index]

cap=cv2.VideoCapture(0)
#face  detection
face_cascade= cv2.CascadeClassifier("hfd.xml")
skip=0
dataset_path='./data/'
face_data=[]
labels=[]
class_id=0 #labels
names={} #record of names with labels
#data preperation
for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		#create mapping btw id and label
		names[class_id]=fx[:-4]
		data_item=np.load(dataset_path+fx)
		print("loaded  "+fx)
		face_data.append(data_item)
		#creating labels for the class
		target= class_id*np.ones((data_item.shape[0],))
		class_id+=1
		labels.append(target)
face_dataset=np.concatenate(face_data,axis=0)
face_labels=np.concatenate(labels,axis=0).reshape((-1,1))
#concatinate data set and lable are training data accepts a matrix
trainset= np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)
while True:
	ret,frame=cap.read()
	if ret== False:
		continue
	faces= face_cascade.detectMultiScale(frame,1.3,5)
	for face in faces[-1:]:
		x,y,w,h=face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
		#Extract (Crop out the required face): region of intrest
		offset=10
		face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
		fs=cv2.resize(face_section,(100,100))
		out=knn(trainset,fs.flatten())
		#display name and rectangle around on pic
		pred_name=names[int(out)]
		cv2.putText(frame, pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,00,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,25,255),2)

	cv2.imshow('Faces',frame)
	key=cv2.waitKey(1)&0xFF
	if key== ord("q"):
		break
cap.release()
cv2.destroyAllWindows()