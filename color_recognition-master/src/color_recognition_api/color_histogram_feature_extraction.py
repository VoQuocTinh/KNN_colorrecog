
from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Hàm xử lý hình ảnh đầu vào để tạo histogram màu cho ảnh cần dự đoán
def color_histogram_of_test_image(test_src_image):

    # Tải hình ảnh
    image = test_src_image

    # Tách kênh màu (Blue, Green, Red)
    chans = cv2.split(image)
    colors = ('b', 'g', 'r')  # Danh sách kênh màu
    features = []             # Danh sách lưu đặc trưng histogram
    feature_data = ''         # Chuỗi lưu giá trị đặc trưng RGB
    counter = 0               # Bộ đếm để theo dõi kênh hiện tại

    for (chan, color) in zip(chans, colors):
        counter += 1

        # Tính histogram cho từng kênh màu
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # Tìm giá trị pixel đỉnh cao nhất trong histogram
        elem = np.argmax(hist)

        # Gán giá trị theo thứ tự màu
        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue  # Dữ liệu RGB

    # Ghi dữ liệu đặc trưng vào tệp "test.data"
    with open('test.data', 'w') as myfile:
        myfile.write(feature_data)


# Hàm tạo histogram màu từ hình ảnh huấn luyện
def color_histogram_of_training_image(img_name):

    # Phát hiện màu sắc từ tên tệp để gắn nhãn cho dữ liệu huấn luyện
    if 'red' in img_name:
        data_source = 'red'
    elif 'yellow' in img_name:
        data_source = 'yellow'
    elif 'green' in img_name:
        data_source = 'green'
    elif 'orange' in img_name:
        data_source = 'orange'
    elif 'white' in img_name:
        data_source = 'white'
    elif 'black' in img_name:
        data_source = 'black'
    elif 'blue' in img_name:
        data_source = 'blue'
    elif 'violet' in img_name:
        data_source = 'violet'

    # Tải hình ảnh
    image = cv2.imread(img_name)

    # Tách kênh màu (Blue, Green, Red)
    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0

    for (chan, color) in zip(chans, colors):
        counter += 1

        # Tính histogram cho từng kênh màu
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # Tìm giá trị pixel đỉnh cao nhất trong histogram
        elem = np.argmax(hist)

        # Gán giá trị theo thứ tự màu
        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue  # Dữ liệu RGB

    # Ghi dữ liệu đặc trưng và nhãn vào tệp "training.data"
    with open('training.data', 'a') as myfile:
        myfile.write(feature_data + ',' + data_source + '\n')


# Hàm huấn luyện, xử lý toàn bộ ảnh trong thư mục dữ liệu huấn luyện
def training():

    # Ảnh huấn luyện màu đỏ
    for f in os.listdir('./training_dataset/red'):
        color_histogram_of_training_image('./training_dataset/red/' + f)

    # Ảnh huấn luyện màu vàng
    for f in os.listdir('./training_dataset/yellow'):
        color_histogram_of_training_image('./training_dataset/yellow/' + f)

    # Ảnh huấn luyện màu xanh lá
    for f in os.listdir('./training_dataset/green'):
        color_histogram_of_training_image('./training_dataset/green/' + f)

    # Ảnh huấn luyện màu cam
    for f in os.listdir('./training_dataset/orange'):
        color_histogram_of_training_image('./training_dataset/orange/' + f)

    # Ảnh huấn luyện màu trắng
    for f in os.listdir('./training_dataset/white'):
        color_histogram_of_training_image('./training_dataset/white/' + f)

    # Ảnh huấn luyện màu đen
    for f in os.listdir('./training_dataset/black'):
        color_histogram_of_training_image('./training_dataset/black/' + f)

    # Ảnh huấn luyện màu xanh dương
    for f in os.listdir('./training_dataset/blue'):
        color_histogram_of_training_image('./training_dataset/blue/' + f)
