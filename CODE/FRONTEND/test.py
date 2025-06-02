from flask import Flask, render_template, redirect, url_for, flash, request, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import pandas as pd
import joblib
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(1), nullable=False)
    mobile = db.Column(db.String(15), nullable=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id  # Store user ID in session
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'danger')
    return render_template('auth.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        age = request.form.get('age')
        gender = request.form.get('gender')
        mobile = request.form.get('mobile')
        
        if len(mobile) != 10 or not mobile.isdigit():
            flash('Mobile number must be exactly 10 digits.', 'danger')
            return render_template('auth.html')

        if User.query.filter_by(email=email).first():
            flash('Email address already in use. Please choose a different one.', 'danger')
            return render_template('auth.html')
        
        if User.query.filter_by(username=username).first():
            flash('Username is already taken. Please choose a different one.', 'danger')
            return render_template('auth.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('auth.html')
        
        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'danger')
            return render_template('auth.html')

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password, age=age, gender=gender, mobile=mobile)
        
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('auth.html')

@app.route('/home')
def home():
    return render_template('home.html')

# Load feature names and models
feature_names = joblib.load('feature_names.pkl')
class_names = {
    0: 'low',
    1: 'high',
    2: 'heavy',
    3: 'normal'
}
decision_tree = joblib.load('decision_tree_model.pkl')
vehicle_traffic_forecasting_model = joblib.load('sarima_model_3h.pkl')

from sklearn.impute import SimpleImputer
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        model_type = request.form['model_type']
        time_str = request.form.get('time')
        car_count = int(request.form.get('car_count', 0))
        bike_count = int(request.form.get('bike_count', 0))
        bus_count = int(request.form.get('bus_count', 0))
        truck_count = int(request.form.get('truck_count', 0))
        day_of_week = request.form.get('day_of_week')

        print(f"Model Type: {model_type}")
        print(f"Time: {time_str}")
        print(f"Car Count: {car_count}")
        print(f"Bike Count: {bike_count}")
        print(f"Bus Count: {bus_count}")
        print(f"Truck Count: {truck_count}")
        print(f"Day of Week: {day_of_week}")

        def time_to_seconds(time_str):
            try:
                time_obj = datetime.strptime(time_str, '%I:%M:%S %p')
                return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
            except ValueError:
                return None
        
        time_in_seconds = time_to_seconds(time_str) if time_str else None
        if time_str and time_in_seconds is None:
            return "Error: Time format should be 'HH:MM:SS AM/PM'"

        input_data = {}
        if model_type == 'vehicle_traffic_forecasting':
            input_data = {
                'Time': [time_in_seconds],
                'Day of the week': [day_of_week]
            }
            print(input_data)
        elif model_type == 'traffic_prediction':
            input_data = {
                'CarCount': [car_count],
                'BikeCount': [bike_count],
                'BusCount': [bus_count],
                'TruckCount': [truck_count],
                'Day of the week': [day_of_week]
            }
        else:
            return "Error: Invalid model type"

        input_df = pd.DataFrame(input_data)
        print(f"Columns before reindexing: {input_df.columns}")
        input_df = input_df.reindex(columns=feature_names)
        print(f"Columns after reindexing: {input_df.columns}")

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        input_df_imputed = pd.DataFrame(imputer.fit_transform(input_df), columns=input_df.columns)

        try:
            if model_type == 'vehicle_traffic_forecasting':
                model = vehicle_traffic_forecasting_model
            elif model_type == 'traffic_prediction':
                model = decision_tree

            prediction = model.predict(input_df_imputed)
            class_name = class_names.get(prediction[0], 'Unknown')
            probabilities = model.predict_proba(input_df_imputed)[0]
        except Exception as e:
            return f"Error in prediction: {str(e)}"

        return render_template('predict.html', prediction={
            'model': model_type.capitalize().replace('_', ' '),
            'class_name': class_name,
            'probabilities': probabilities
        })

    return render_template('predict.html')


@app.route('/get_traffic/<float:lat>/<float:lon>')
def get_traffic(lat, lon):
    # Mock response for demonstration; replace with actual logic
    traffic_condition = 'Light'  # Example response, replace with real logic
    
    return jsonify({
        'traffic_condition': traffic_condition
    })

@app.route('/map_view')
def map_view():
    return render_template('map.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)  # Remove user ID from session
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create the database tables
    app.run(debug=True)





# from flask import Flask, render_template, redirect, url_for, flash, request, session, jsonify
# from flask_sqlalchemy import SQLAlchemy
# from flask_bcrypt import Bcrypt
# import pandas as pd
# import joblib
# from datetime import datetime

# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'your_secret_key'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

# db = SQLAlchemy(app)
# bcrypt = Bcrypt(app)

# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(150), unique=True, nullable=False)
#     email = db.Column(db.String(150), unique=True, nullable=False)
#     password = db.Column(db.String(150), nullable=False)
#     age = db.Column(db.Integer, nullable=False)
#     gender = db.Column(db.String(1), nullable=False)
#     mobile = db.Column(db.String(15), nullable=False)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/about')
# def about():
#     return render_template('about.html')

# @app.route('/contact')
# def contact():
#     return render_template('contact.html')

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         email = request.form.get('email')
#         password = request.form.get('password')

#         user = User.query.filter_by(email=email).first()
#         if user and bcrypt.check_password_hash(user.password, password):
#             session['user_id'] = user.id  # Store user ID in session
#             # flash('Login successful!', 'success')
#             return redirect(url_for('home'))
#         else:
#             flash('Invalid email or password.', 'danger')
#     return render_template('auth.html')

# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         username = request.form.get('username')
#         email = request.form.get('email')
#         password = request.form.get('password')
#         confirm_password = request.form.get('confirm_password')
#         age = request.form.get('age')
#         gender = request.form.get('gender')
#         mobile = request.form.get('mobile')
        
#         if len(mobile) != 10 or not mobile.isdigit():
#             flash('Mobile number must be exactly 10 digits.', 'danger')
#             return render_template('auth.html')

#         if User.query.filter_by(email=email).first():
#             flash('Email address already in use. Please choose a different one.', 'danger')
#             return render_template('auth.html')
        
#         if User.query.filter_by(username=username).first():
#             flash('Username is already taken. Please choose a different one.', 'danger')
#             return render_template('auth.html')

#         if password != confirm_password:
#             flash('Passwords do not match.', 'danger')
#             return render_template('auth.html')
        
#         if len(password) < 8:
#             flash('Password must be at least 8 characters long.', 'danger')
#             return render_template('auth.html')

#         hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
#         new_user = User(username=username, email=email, password=hashed_password, age=age, gender=gender, mobile=mobile)
        
#         db.session.add(new_user)
#         db.session.commit()

#         flash('Registration successful! You can now log in.', 'success')
#         return redirect(url_for('login'))
#     return render_template('auth.html')

# @app.route('/home')
# def home():
#     return render_template('home.html')




# # Load feature names
# feature_names = joblib.load('feature_names.pkl')

# # Define class names mapping
# class_names = {
#     0: 'low',
#     1: 'high',
#     2: 'heavy',
#     3: 'normal'
# }

# # Load the trained models
# decision_tree = joblib.load('decision_tree_model.pkl')
# vehicle_traffic_forecasting_model = joblib.load('sarima_model_3h.pkl')

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         model_type = request.form['model_type']
#         time_str = request.form['time']
#         car_count = int(request.form['car_count'])
#         bike_count = int(request.form['bike_count'])
#         bus_count = int(request.form['bus_count'])
#         truck_count = int(request.form['truck_count'])
#         day_of_week = int(request.form['day_of_week'])

#         def time_to_seconds(time_str):
#             try:
#                 time_obj = datetime.strptime(time_str, '%I:%M:%S %p')
#                 return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
#             except ValueError:
#                 return None
        
#         time_in_seconds = time_to_seconds(time_str)
#         if time_in_seconds is None:
#             return "Error: Time format should be 'HH:MM:SS AM/PM'"

#         input_data = {
#             'Time': [time_in_seconds],
#             'CarCount': [car_count],
#             'BikeCount': [bike_count],
#             'BusCount': [bus_count],
#             'TruckCount': [truck_count],
#             'Day of the week': [day_of_week]
#         }

#         input_df = pd.DataFrame(input_data)
#         input_df = input_df.reindex(columns=feature_names)

#         try:
#             if model_type == 'vehicle_traffic_forecasting':
#                 model = vehicle_traffic_forecasting_model
#             elif model_type == 'traffic_prediction':
#                 model = decision_tree
#             else:
#                 return "Error: Invalid model type"

#             prediction = model.predict(input_df)
#             class_name = class_names.get(prediction[0], 'Unknown')
#             probabilities = model.predict_proba(input_df)[0]
#         except Exception as e:
#             return f"Error in prediction: {str(e)}"

#         return render_template('predict.html', prediction={
#             'model': model_type.capitalize().replace('_', ' '),
#             'class_name': class_name,
#             'probabilities': probabilities
#         })

#     return render_template('predict.html')

# @app.route('/get_traffic/<float:lat>/<float:lon>')
# def get_traffic(lat, lon):
#     # Mock response for demonstration; you should replace with actual logic
#     traffic_condition = 'Light'  # or 'Heavy', based on your logic
    
#     return jsonify({
#         'traffic_condition': traffic_condition
#     })

# # @app.route('/predict')
# # def predict():
# #     return render_template('predict.html')

# @app.route('/logout')
# def logout():
#     session.pop('user_id', None)  # Remove user ID from session
#     flash('You have been logged out.', 'info')
#     return redirect(url_for('login'))

# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()  # Create the database tables
#     app.run(debug=True)
