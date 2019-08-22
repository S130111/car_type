from keras.preprocessing.image import img_to_array
from keras.models import load_model
from car_color.color import Color
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
#[轿车，SUV,MPV,面包车，出租车]
car_type = ['Car','SUV',"MPV","MinNiBus","Taxi"]


for path,dirs,files in os.walk(r'C:\Users\ThinkPad\Desktop\1'):
	for file in files:
		car_type_list, car_class_list= [],[]
		image = cv2.imread(os.path.join(path,file))
		output = imutils.resize(image, width=400)

		image = cv2.resize(image, (96, 96))
		image = image.astype("float") / 255.0
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)

		print("[INFO] 读取模型与标签.....")
		model = load_model('car_label.model')
		mlb = pickle.loads(open("mlb.pickle", "rb").read())

		print("[INFO] 识别图片.....")
		proba = model.predict(image)[0]
		idxs = np.argsort(proba)[::-1]

		#
		for (i, j) in enumerate(idxs):
			if mlb.classes_[j] in car_type:
				car_type_list.append((mlb.classes_[j], proba[j] * 100))
			else:
				if len(car_class_list) == 4:
					continue
				else:
					car_class_list.append((mlb.classes_[j], proba[j] * 100))
		print("车辆种类")
		for (label,p) in car_class_list:
			print("{}: {:.2f}%".format(label, p))
		print("车型")
		for (label,p) in car_type_list:
			print("{}: {:.2f}%".format(label, p))
		car_color = Color(os.path.join(path,file))
		print("车颜色：",car_color)

			# print("{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100))
		# 	# build the label and draw the label on the image
		# 	label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
		# 	cv2.putText(output, label, (10, (i * 30) + 25),
		# 		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

		# show the probabilities for each of the individual labels
		# for (label, p) in zip(mlb.classes_, proba):
		# 	print("{}: {:.2f}%".format(label, p * 100))
		# print("+++++++++++++++++++++++++++")
		# show the output image
		# cv2.imshow("Output", output)
		# cv2.waitKey(0)