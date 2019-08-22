from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os
import numpy as np


model = load_model(r'D:\wuxisuo\che_liang_te_zheng\car_color\car_color_4.h5')
color_list = ["黑色",'蓝色','棕色','香槟色','灰色','绿色','橙色','紫色','红色','白色','黄色']
# test_dir =r'C:\Users\ThinkPad\Desktop\111\white'
# w = 0
# h = 0
# v = 0
# for root, dirs, files in os.walk(test_dir):
#     for jpg_path in files:
def Color(path):
    img = image.load_img(path, target_size=(300, 300))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255
    # model.predict(img_tensor)
    i = model.predict(img_tensor)
    # print(i)
    predict = np.argmax(i, axis=1)
    car_color = color_list[predict[0]]
    return  car_color
        # print(color_list[predict[0]])
        # if color_list[predict[0]] == "白色":
        #     w += 1
        # else:
        #     h +=1
        # if v == 100:
        #     print(w)
        #     print(h)
        # v += 1

        # print(np.where(i == np.max(i))[1])
        # print(i[1])
        # j = (int(i[1])+2) % 4     # label 装换
        # with open('result.txt', "a") as f:
        #     f.write(str(j) + '#' + jpg_path+'\n')