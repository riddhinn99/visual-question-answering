import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from model_new import NewModel
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'images/'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
	return render_template('login.html')

@app.route('/success/<question>,<test_image>')
def success(question,test_image):
	test_image = 'http://127.0.0.1:5000/uploads/' + test_image
	return render_template('Answer.html', quest=question, image=test_image)


@app.route('/uploads/<test_image>')
def send_file(test_image):
    return send_from_directory(UPLOAD_FOLDER, test_image)


@app.route('/login',methods = ['POST', 'GET'])
def login():
	ques = request.form['question']
	file = request.files['file']
	        # if user does not select file, browser also
	        # submit an empty part without filename
	if file.filename == '':
		return 'NO FILE'
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

	image_path = 'images/'
	image_path = image_path + file.filename
	question = ques
	obj = NewModel()
	ans = obj.predict_in_class('sample_data_new/11.jpg', 'is this a car')
	return render_template('login.html', prediction_text = ans)
	#return redirect(url_for('success',question = ques, test_image= filename))
		
if __name__ == "__main__":
	app.run(debug = True)
