import cv2
import numpy as np
import onnxruntime as ort

# Load mô hình YOLOv8
session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])

# Đọc ảnh
def Image_read(image_path):
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    return [image, h, w]

# Tiền xử lý ảnh cho YOLOv8
def Preproccess_image(image):
    image_resized = cv2.resize(image, (960, 960))
    image_tensor = np.transpose(image_resized[:, :, ::-1].astype(np.float32) / 255.0, (2, 0, 1))[None]
    
    return image_tensor

# Dự đoán bounding boxes
def Detect(session, image_tensor, h, w):
    outputs = session.run(None, {session.get_inputs()[0].name: image_tensor})[0].squeeze()
    x_c, y_c, w_b, h_b, scores = outputs

    # Lọc box theo threshold
    mask = scores > 0.05
    x1, y1 = (x_c - w_b / 2)[mask] * (w / 960), (y_c - h_b / 2)[mask] * (h / 960)
    x2, y2 = (x_c + w_b / 2)[mask] * (w / 960), (y_c + h_b / 2)[mask] * (h / 960)
    filtered_scores = scores[mask]

    if len(filtered_scores) > 0:
        max_idx = np.argmax(filtered_scores)
        x1, y1, x2, y2 = map(int, [x1[max_idx], y1[max_idx], x2[max_idx], y2[max_idx]])

        # Vẽ bounding box lên ảnh gốc
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Hiển thị ảnh kết quả

    image = cv2.resize(image, (540, 960))

    return image