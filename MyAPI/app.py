from flask import Flask, render_template, request, jsonify, session, url_for , redirect
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import os
import numpy as np
import cv2

app = Flask(__name__)

model = YOLO("weights/best.pt") 

# Các lớp đối tượng trong mô hình YOLOv8
classes = ['c-dom-la', 'c-khoe-manh']  # Thay thế bằng danh sách các lớp thực tế

# Hàm tiền xử lý ảnh
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
])

def draw_bbox(image_path, results):	
    # Load the image	
    image = cv2.imread(image_path)	
    	
    # Iterate over the results	
    for result in results:	
        bbox = result["bbox"]	
        class_name = result["class"]	
        confidence = result["confidence"]	
        	
        # Unpack the bounding box coordinates	
        xmin, ymin, xmax, ymax = bbox	
        	
        # Convert float coordinates to integers	
        xmin = int(xmin)	
        ymin = int(ymin)	
        xmax = int(xmax)	
        ymax = int(ymax)	
        	
        # Draw the bounding box rectangle on the image	
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)	
        	
        # Create a text label	
        label = f"{class_name}: {confidence:.2f}"	
        	
        # Draw the label background rectangle	
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)	
        cv2.rectangle(image, (xmin, ymin - label_size[1]), (xmin + label_size[0], ymin + base_line), (0, 255, 0), cv2.FILLED)	
        	
        # Draw the text label	
        cv2.putText(image, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)	
    	
    # Save the drawn image	
    cv2.imwrite(f"static/test/predict_{os.path.basename(image_path)}", image)
    return f"static/test/predict_{os.path.basename(image_path)}"

# Route giao diện chính
@app.route('/')
def index():
    return render_template('index.html')

# Route xử lý phân loại đối tượng
@app.route('/detect', methods=['POST'])
def detect():
    # Kiểm tra xem có file hình ảnh được tải lên không
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    # Kiểm tra định dạng tệp hợp lệ
    if file.filename == '':
        # return jsonify({'error': 'No file selected'})
        return render_template("index.html", error = "No file selected")

    if file:
        # Lưu file ảnh tải lên vào thư mục tạm
        filename = secure_filename(file.filename)
        if not os.path.isdir("temp"):
            os.makedirs('temp')
        temp_path = 'temp/' + filename
        file.save(temp_path)
        outputs = model(temp_path)
        

        # Xử lý kết quả đầu ra
        results = []
        for output in outputs:
            data = output.boxes
            for boxes, cls, conf in zip(data.xyxy, data.cls, data.conf):
                class_idx = int(cls)
                class_label = classes[class_idx] 
                confidence = float(conf)

                # Lấy tọa độ bounding box
                bbox = boxes[:4].tolist()

                # Thêm thông tin vào kết quả
                results.append({
                    'class': class_label,
                    'confidence': confidence,
                    'bbox': bbox
                })
        
        predict_img_name = draw_bbox(temp_path, results)

        # Trả về kết quả object detection
        data =  {'result': results, 'image': predict_img_name}
        return render_template("index.html", data = data)

        

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port='6868')
    app.run(debug=True)
