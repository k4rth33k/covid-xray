import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import cv2
from flask import Flask, request, jsonify, Blueprint, render_template
from timeit import default_timer as timer
import os


app = Flask(__name__, template_folder='./templates')

TF_SESSION = None
OUTPUT_TENSOR = None
IN_TENSOR = None

def download_model():
	pass

def init_model():
	global TF_SESSION
	global OUTPUT_TENSOR
	global IN_TENSOR


	tf.keras.backend.set_learning_phase(0)
	TF_SESSION=tf.InteractiveSession()

	model_name = os.listdir('../azure/model/')[0]
	# model_name = 'tf_model_31_03_2020__12_38_09 (1).pb'
	# model_path = f'./model/{model_name}'
	model_path = f'../azure/model/{model_name}'


	with tf.gfile.GFile(model_path, "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	with TF_SESSION.graph.as_default() as graph:
		tf.import_graph_def(graph_def)

	in_tensor = ''
	out_tensor = ''

	for op in graph.get_operations():
		for inp in op.inputs:
			if 'input' in inp.name:
				in_tensor = inp.name

			if 'Sigmoid' in op.name:
				out_tensor = inp.name

	OUTPUT_TENSOR = TF_SESSION.graph.get_tensor_by_name(out_tensor)
	TF_SESSION.run(tf.global_variables_initializer())
	IN_TENSOR = in_tensor



@app.route('/',methods=['GET'])
def home():
	return render_template("index.html")

@app.route('/api',methods=['POST'])
def api():
	# print(request.files)
	start = timer()
	file = request.files['image'].read()

	np_bytes = np.fromstring(file, np.uint8)
	img = cv2.imdecode(np_bytes, cv2.COLOR_BGR2RGB)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	img = cv2.resize(img, (224, 224))
	img = img.astype('float')
	img = img / 255.0
	img = np.array([img])

	try:
		predictions = TF_SESSION.run(OUTPUT_TENSOR, {IN_TENSOR: img}).sum().item()
		print(predictions)
		label = 'Normal' if predictions < 0.5 else 'COVID'
		end = timer()

		resp_time = end - start

		return jsonify({'API vesion' : '1',
						'result' :{ 'confidence' : predictions,
						'predicted_label' : label },
						'Response time' : f'{resp_time} seconds',
						'disclaimer' : 'This API does not claim any medical correctness for the rendered results'})
	except Exception as e:
		return jsonify({'error' : str(e)})

if __name__ == '__main__':
	init_model()
	port = 80
	app.run(debug=True, host='0.0.0.0', port=port)







