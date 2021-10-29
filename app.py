from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np 
import os
import model as md
import predict
import cv2

app = Flask(__name__)

file_names = 'coco.names'
file_weight = 'yolov3.weights'
file_cfg = 'yolov3.cfg'

file_ct_names = 'yolo.names'
file_ct_weight = 'yolov4-custom.weights'
file_ct_cfg = 'yolov4-custom.cfg'
model = md.Model(file_names, file_weight, file_cfg)
model2 = md.Model(file_ct_names, file_ct_weight, file_ct_cfg)

@app.route('/')
@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/slideshow')
def slideshow():
	return render_template('slideshow2.html')

@app.route('/loadimage')
def loadimage():
	return render_template('image.html')

@app.route('/loadimage2')
def loadimage2():
	return render_template('image2.html')

@app.route('/detect_object', methods=['POST'])
def detectObject():
	if request.method == 'POST':
		#lấy files từ request
		files = request.files.getlist("files")
		imgurls = []
		titles = []
		for file in files:
			#chuyển sang dữ liệu ảnh
			img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
			res, title = predict.predict(model, img)
			res_name = "result_"+ file.filename
			imgurls.append("../static/result_image/"+res_name)
			titles.append(title)
			destination = os.path.join(os.path.dirname(os.path.realpath(__file__)),"static/result_image",res_name)
			cv2.imwrite(destination, res)
		
		img_info = zip(imgurls, titles)
	return render_template('result.html', images = img_info)

@app.route('/detect_object2', methods=['POST'])
def detectObject2():
	if request.method == 'POST':
		files = request.files.getlist("files")
		imgurls = []
		titles = []		
		for file in files:
			img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
			res, title = predict.predict(model2, img)
			res_name = "result_"+ file.filename
			imgurls.append("../static/result_image/"+res_name)
			titles.append(title)
			destination = os.path.join(os.path.dirname(os.path.realpath(__file__)),"static/result_image",res_name)
			cv2.imwrite(destination, res)
		
		img_info = zip(imgurls, titles)
	return render_template('result2.html', images = img_info)

if __name__ == '__main__':
	app.run(debug = True)