import cv2
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os

# Mở camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Lỗi: Không thể mở camera.")
    exit()

# Đọc một khung hình từ webcam
ret, frame = cap.read()
prediction = 'n.a.'  # Kết quả dự đoán ban đầu (chưa xác định)

# Đường dẫn tới dữ liệu huấn luyện
PATH = './training.data'

# Kiểm tra xem dữ liệu huấn luyện có tồn tại không
if os.path.isfile(PATH) and os.access(PATH, os.R_OK):  # Nếu tệp tồn tại và có thể đọc
    print('Dữ liệu huấn luyện đã sẵn sàng, bộ phân loại đang tải...')
else:
    print('Dữ liệu huấn luyện đang được tạo...')
    open('training.data', 'w')  # Tạo tệp rỗng `training.data`
    color_histogram_feature_extraction.training()  # Gọi hàm huấn luyện để tạo dữ liệu
    print('Dữ liệu huấn luyện đã sẵn sàng, bộ phân loại đang tải...')

# Vòng lặp đọc khung hình từ webcam và phân loại màu
while True:
    ret, frame = cap.read()  # Đọc một khung hình từ webcam

    if not ret:
        print("Lỗi: Không thể đọc khung hình từ webcam.")
        break

    # Hiển thị kết quả dự đoán trên khung hình
    cv2.putText(
        frame,
        'Prediction: ' + prediction,  # Thêm văn bản chứa dự đoán
        (15, 45),  # Tọa độ hiển thị văn bản
        cv2.FONT_HERSHEY_PLAIN,  # Phông chữ
        3,  # Kích thước chữ
        (200, 200, 200),  # Màu sắc (RGB)
    )

    cv2.imshow('color classifier', frame)  # Hiển thị khung hình với tên "color classifier"

    # Trích xuất đặc trưng màu từ khung hình
    color_histogram_feature_extraction.color_histogram_of_test_image(frame)

    # Dự đoán màu sắc từ khung hình hiện tại
    prediction = knn_classifier.main('training.data', 'test.data')

    # Nếu người dùng nhấn phím 'q', thoát khỏi vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng các cửa sổ hiển thị
cap.release()
cv2.destroyAllWindows()
