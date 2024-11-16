import csv
import random
import math
import operator
import cv2

# Tính khoảng cách Euclid giữa hai điểm dữ liệu
def calculateEuclideanDistance(variable1, variable2, length):
    distance = 0
    for x in range(length):
        distance += pow(variable1[x] - variable2[x], 2)  # Bình phương hiệu giữa các đặc trưng
    return math.sqrt(distance)  # Căn bậc hai tổng bình phương

# Tìm k láng giềng gần nhất
def kNearestNeighbors(training_feature_vector, testInstance, k):
    distances = []  # Danh sách lưu khoảng cách
    length = len(testInstance)  # Số lượng đặc trưng trong một mẫu
    for x in range(len(training_feature_vector)):
        dist = calculateEuclideanDistance(testInstance, training_feature_vector[x], length)  # Tính khoảng cách
        distances.append((training_feature_vector[x], dist))  # Ghi lại mẫu và khoảng cách
    distances.sort(key=operator.itemgetter(1))  # Sắp xếp theo khoảng cách (tăng dần)
    neighbors = []  # Lưu k láng giềng gần nhất
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

# Lấy kết quả dự đoán từ láng giềng
def responseOfNeighbors(neighbors):
    all_possible_neighbors = {}  # Lưu số phiếu cho mỗi nhãn
    for x in range(len(neighbors)):
        response = neighbors[x][-1]  # Nhãn của mẫu láng giềng (vị trí cuối cùng)
        if response in all_possible_neighbors:
            all_possible_neighbors[response] += 1  # Tăng phiếu nếu nhãn đã tồn tại
        else:
            all_possible_neighbors[response] = 1  # Thêm nhãn mới
    # Sắp xếp nhãn theo số phiếu giảm dần
    sortedVotes = sorted(all_possible_neighbors.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]  # Trả về nhãn có số phiếu cao nhất

# Đọc dữ liệu đặc trưng từ tệp và phân chia thành vector huấn luyện và kiểm tra
def loadDataset(
    filename,
    filename2,
    training_feature_vector=[],  # Vector đặc trưng huấn luyện
    test_feature_vector=[],  # Vector đặc trưng kiểm tra
    ):
    with open(filename) as csvfile:  # Mở tệp dữ liệu huấn luyện
        lines = csv.reader(csvfile)
        dataset = list(lines)  # Đọc toàn bộ nội dung tệp
        for x in range(len(dataset)):  # Duyệt qua từng mẫu dữ liệu
            for y in range(3):  # Duyệt qua các đặc trưng RGB
                dataset[x][y] = float(dataset[x][y])  # Chuyển đổi sang kiểu số thực
            training_feature_vector.append(dataset[x])  # Thêm vào vector huấn luyện

    with open(filename2) as csvfile:  # Mở tệp dữ liệu kiểm tra
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(3):
                dataset[x][y] = float(dataset[x][y])  # Chuyển đổi sang kiểu số thực
            test_feature_vector.append(dataset[x])  # Thêm vào vector kiểm tra

# Hàm chính, thực hiện quá trình phân loại
def main(training_data, test_data):
    training_feature_vector = []  # Vector đặc trưng huấn luyện
    test_feature_vector = []  # Vector đặc trưng kiểm tra
    loadDataset(training_data, test_data, training_feature_vector, test_feature_vector)  # Tải dữ liệu
    classifier_prediction = []  # Danh sách kết quả dự đoán
    k = 3  # Giá trị k của KNN (số láng giềng gần nhất)
    for x in range(len(test_feature_vector)):  # Duyệt qua từng mẫu kiểm tra
        neighbors = kNearestNeighbors(training_feature_vector, test_feature_vector[x], k)  # Tìm láng giềng
        result = responseOfNeighbors(neighbors)  # Dự đoán nhãn từ láng giềng
        classifier_prediction.append(result)  # Lưu kết quả
    return classifier_prediction[0]  # Trả về dự đoán đầu tiên
