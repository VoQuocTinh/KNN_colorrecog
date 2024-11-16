import cv2
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os
import os.path
import sys

# Đọc ảnh đầu vào
try:
    source_image = cv2.imread(sys.argv[1])  # Đọc ảnh từ tham số dòng lệnh
except:
    source_image = cv2.imread('black_cat.jpg')  # Sử dụng ảnh mặc định nếu không có tham số
prediction = 'n.a.'  # Giá trị dự đoán ban đầu (chưa xác định)

# Kiểm tra xem dữ liệu huấn luyện đã sẵn sàng hay chưa
PATH = './training.data'

if os.path.isfile(PATH) and os.access(PATH, os.R_OK):  # Kiểm tra tệp huấn luyện có tồn tại và có thể đọc không
    print('Dữ liệu huấn luyện đã sẵn sàng, bộ phân loại đang được tải...')
else:
    print('Dữ liệu huấn luyện đang được tạo...')
    open('training.data', 'w')  # Tạo tệp `training.data` rỗng
    color_histogram_feature_extraction.training()  # Gọi hàm huấn luyện để tạo dữ liệu huấn luyện
    print('Dữ liệu huấn luyện đã sẵn sàng, bộ phân loại đang được tải...')

# Lấy kết quả dự đoán
color_histogram_feature_extraction.color_histogram_of_test_image(source_image)  # Trích xuất đặc trưng từ ảnh kiểm tra
prediction = knn_classifier.main('training.data', 'test.data')  # Dự đoán bằng KNN
print('Màu được phát hiện là:', prediction)

# Hiển thị kết quả dự đoán trên ảnh
cv2.putText(
    source_image,
    'Prediction: ' + prediction,  # Thêm kết quả dự đoán vào ảnh
    (15, 45),  # Tọa độ của văn bản
    cv2.FONT_HERSHEY_PLAIN,  # Phông chữ
    3,  # Kích thước phông
    200,  # Màu sắc (thang xám)
)

# Hiển thị ảnh kèm kết quả phân loại
cv2.imshow('color classifier', source_image)  # Hiển thị ảnh với tên "color classifier"
cv2.waitKey(0)  # Chờ người dùng nhấn phím bất kỳ để đóng cửa sổ
