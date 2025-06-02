from flask import Flask, render_template, redirect, url_for, flash, request, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import pandas as pd
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
from flask import jsonify
from statsmodels.tsa.statespace.sarimax import SARIMAX

sarima_fit = joblib.load('sarima_model_3h.pkl')

# Define a mapping for days of the week
day_of_week_mapping = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}

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

# Load and preprocess data
data = pd.read_csv('TrafficTwoMonth.csv')  # Replace with your actual data file path
# Convert Date and Time to datetime
data['Time'] = pd.to_datetime(data['Time'], format='%I:%M:%S %p').dt.time
# Encode categorical variables
label_encoder = LabelEncoder()
data['Day of the week'] = label_encoder.fit_transform(data['Day of the week'])
data['Traffic Situation'] = label_encoder.fit_transform(data['Traffic Situation'])
# Drop any remaining non-numeric columns if any
data = data.select_dtypes(include=[np.number])
data=data.drop(["Date","Total"],axis=1)
print(data.columns)
# Define features and target variable
X = data.drop(['Traffic Situation'], axis=1)
y = data['Traffic Situation']
feature_names = X.columns.tolist()

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Normalize the resampled features
scaler = StandardScaler()  # or MinMaxScaler()
X_resampled_normalized = scaler.fit_transform(X_resampled)

# Split the balanced and normalized dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled_normalized, y_resampled, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Classifier
decision_tree = DecisionTreeClassifier()
# Ensure that the model is trained with DataFrames having feature names
X_train_df = pd.DataFrame(X_train, columns=feature_names)
decision_tree.fit(X_train_df, y_train)

# @app.route('/predict1', methods=['GET', 'POST'])
# def predict1():
#     if request.method == 'POST':
#         forecast_hours = int(request.form.get('forecast_hours', 24))

#         print(f"Starting forecast for {forecast_hours} hours.")
#         # Load the data
#         file_path = 'traffic.csv'
#         data = pd.read_csv(file_path)

#         # Parse the DateTime column to datetime objects and set it as the index
#         data['DateTime'] = pd.to_datetime(data['DateTime'])
#         data.set_index('DateTime', inplace=True)

#         # Downsample the data to every 3 hours
#         data_3h = data.resample('3H').sum()

#         steps = forecast_hours // 3
#         # Forecast
#         arima_forecast = sarima_fit.forecast(steps=steps)
        
#         arima_forecast = [str(i) for i in arima_forecast]

#         time_list = {"3": 1, "6": 2, "9": 3, "12": 4, "15": 5, "18": 6, "21": 7, "24": 8}
#         time = time_list[str(forecast_hours)]
#         print(f"SARIMA Forecast for next {forecast_hours} hours (3-hour intervals):\n{arima_forecast}\n")

#         return render_template('predict1.html', forecast=f'Forecasting Completed for {forecast_hours} hours', arima_forecast = arima_forecast, time = time)
#     return render_template('predict1.html')


@app.route('/predict1', methods=['GET', 'POST'])
def predict1():
    if request.method == 'POST':
        forecast_hours = int(request.form.get('forecast_hours', 24))

        print(f"Starting forecast for {forecast_hours} hours.")
        # Load the data
        file_path = 'traffic.csv'
        data = pd.read_csv(file_path)

        # Parse the DateTime column to datetime objects and set it as the index
        data['DateTime'] = pd.to_datetime(data['DateTime'])
        data.set_index('DateTime', inplace=True)

        # Downsample the data to hourly frequency
        data_3h = data.resample('H').sum()

        # Select the 'Vehicles' column for forecasting
        if 'Vehicles' not in data_3h.columns:
            return "Error: 'Vehicles' column not found in the dataset."

        series = data_3h['Vehicles']

        # # Define and fit the SARIMA model on the selected column
        # sarima_model = SARIMAX(series, 
        #                        order=(1, 1, 1), 
        #                        seasonal_order=(1, 1, 1, 24)) 
        # sarima_fit = sarima_model.fit(disp=False)
        sarima_model = SARIMAX(series, 
                               order=(1, 1, 1), 
                               seasonal_order=(1, 0, 0, 24))  # Simplified seasonal order
        sarima_fit = sarima_model.fit(disp=False, maxiter=50) 

        # Forecast
        steps = forecast_hours
        arima_forecast = sarima_fit.forecast(steps=steps)

        decimal_places = 0
        arima_forecast = [str(round(i, decimal_places)) for i in arima_forecast]

        time_list = {
            "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
            "9": 9, "10": 10, "11": 11, "12": 12, "13": 13, "14": 14, "15": 15, "16": 16,
            "17": 17, "18": 18, "19": 19, "20": 20, "21": 21, "22": 22, "23": 23, "24": 24
        }

        time = time_list.get(str(forecast_hours), 0)
        print(time)
        print(f"SARIMA Forecast for next {forecast_hours} hours:\n{arima_forecast}\n")

        return render_template('predict1.html', forecast=f'Forecasting Completed for {forecast_hours} hours', arima_forecast=arima_forecast, time=time)
    return render_template('predict1.html')

@app.route('/predict2', methods=['GET', 'POST'])
def predict2():
    if request.method == 'POST':
        car_count = int(request.form.get('car_count', 0))
        bike_count = int(request.form.get('bike_count', 0))
        bus_count = int(request.form.get('bus_count', 0))
        truck_count = int(request.form.get('truck_count', 0))
        day_of_week = request.form.get('day_of_week', '')

        day_of_week_encoded = day_of_week_mapping[day_of_week]

        # Prepare the input data with the correct order of features
        input_data = {
            'Day of the week': day_of_week_encoded,  # Use mapped value
            'CarCount': car_count,
            'BikeCount': bike_count,
            'BusCount': bus_count,
            'TruckCount': truck_count
        }

        # Create DataFrame with feature names matching the training data
        input_df = pd.DataFrame([input_data])  # Input data needs to be passed as a list of dictionaries
        
        # Ensure the DataFrame columns match the training features
        input_df = input_df[feature_names]  # Select only the columns in feature_names and in the correct order
        
        # Apply the same scaling
        input_df_scaled = scaler.transform(input_df)

        # Convert back to DataFrame to keep feature names
        input_df_scaled = pd.DataFrame(input_df_scaled, columns=feature_names)

        # Predict using the Decision Tree model
        prediction = decision_tree.predict(input_df_scaled)
        class_name = label_encoder.inverse_transform(prediction)[0]  # Use inverse transform to get original class name
        probabilities = decision_tree.predict_proba(input_df_scaled)[0]

        return render_template('predict2.html', class_name = class_name, probabilities = probabilities)
    return render_template('predict2.html')




@app.route('/get_traffic/<float:lat>/<float:lon>')
def get_traffic(lat, lon):
    # Randomly select a traffic condition
    traffic_conditions = ['Heavy', 'Low', 'High', 'Normal']
    traffic_condition = random.choice(traffic_conditions)
    
    # Return the selected traffic condition as JSON
    return jsonify({'traffic_condition': traffic_condition})


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
