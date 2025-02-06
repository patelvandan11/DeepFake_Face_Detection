from flask import Flask, request, jsonify,render_template

app = Flask(__name__)

@app.route('/', endpoint='home')
def home():
    return render_template('index.html')
@app.route('/about')
def about():
    return render_template('Aboutus.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/support')
def support():
    return render_template('support.html')
    


if __name__ == '__main__':
    app.run(debug=True)