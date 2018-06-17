import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
# 使用keras的sequential模型，是某种卷积神经网络，没找到中文名
from keras.models import Sequential
# conv是卷积核，maxpooling是池化核，用来去掉Feature Map中不重要的样本
from keras.layers.convolutional import Conv2D, MaxPooling2D
# flatten将最后的池化层压平成一位数组用来全连接，dense全连接层
from keras.layers.core import Flatten, Dense
from helpers import resize_to_fit

LETTER_IMAGES_FOLDER = "extracted_letter_images"
MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"

# 保存字母图和对应的文字标签
data = []
labels = []

# 循环输入训练字母图，有标签训练，准备好训练集
for image_file in paths.list_files(LETTER_IMAGES_FOLDER):
    # 读取图片并转换为灰度图，然后调整图片大小20*20像素
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = resize_to_fit(image, 20, 20)
    # 添加图像通道，为后序卷积提取feature_map做准备
    image = np.expand_dims(image, axis=2)
    # 提取图片所在文件夹的名字，该名字代表字母图的内容
    label = image_file.split(os.path.sep)[-2]
    # 保存字母图和对应的文字标签
    data.append(image)
    labels.append(label)

# 将图片数据转换到区间[0,1]，数据结构使用ndarray
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
# 划分训练集，验证集，训练/验证比例test_size设定为0.25
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)
# 标签二值化预处理
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)
# 保存标签(序列化数据存储)
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)

# 构造神经网络，按照典型的卷积神经网络进行两次卷积、池化，并一维序列化，全连接，输出
model = Sequential()
# 第一次卷积、池化，第一层要使用input_shape参数，此处表示20*20的灰度图
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# 第二次卷积、池化
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# 全连接隐含层的结点设定为500个
model.add(Flatten())
model.add(Dense(500, activation="relu"))
# 输出层设定有32个结点，正好对应需要识别的每一个字母或数字
model.add(Dense(32, activation="softmax"))
# 编译模型，标准步骤之一，不清楚用途
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# 训练神经网络，validation_data设定验证集，每批32个数据，总共10轮
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=10, verbose=1)
# 保存模型(使用模型自带函数)
model.save(MODEL_FILENAME)