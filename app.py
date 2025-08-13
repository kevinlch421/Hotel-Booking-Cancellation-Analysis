import os
import pickle
import pandas as pd
from flask import Flask, render_template, request
from datetime import datetime
from jinja2 import StrictUndefined

# Create instance of the class
app = Flask(__name__)
app.jinja_env.undefined = StrictUndefined  # Better error messages for templates

# Configuration
MODEL_DIR = "models"

# Load artifacts
try:
    with open(os.path.join(MODEL_DIR, "minmax_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    
    with open(os.path.join(MODEL_DIR, "gradient_boosting_hotel_cancellation.pkl"), "rb") as f:
        model = pickle.load(f)
    
    # Define feature columns
    FEATURE_COLUMNS = [
        "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
        "adults", "children", "babies", "previous_cancellations",
        "previous_bookings_not_canceled", "booking_changes",
        "days_in_waiting_list", "adr", "required_car_parking_spaces",
        "total_of_special_requests",
        "dummy_Resort Hotel", "dummy_2016", "dummy_2017", "dummy_1",
        "dummy_FB", "dummy_HB", "dummy_SC", "dummy_Complementary",
        "dummy_Corporate", "dummy_Direct", "dummy_Groups",
        "dummy_Offline TA/TO", "dummy_Online TA", "dummy_GDS",
        "dummy_TA/TO", "dummy_Undefined", "dummy_B", "dummy_C", "dummy_D",
        "dummy_E", "dummy_F", "dummy_G", "dummy_H", "dummy_L", "dummy_P",
        "dummy_Non Refund", "dummy_Refundable", "dummy_Group",
        "dummy_Transient", "dummy_Transient-Party",
        "dummy_Spring", "dummy_Summer", "dummy_Winter"
    ]
    
    print("Model artifacts loaded successfully")
    
except Exception as e:
    print(f"Error loading model artifacts: {str(e)}")
    # Handle error appropriately
    FEATURE_COLUMNS = []
    model = None
    scaler = None

# Test route
@app.route('/test')
def test():
    return render_template("test.html", now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Favicon route
@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

# Main index route
@app.route("/", methods=["GET"])
def index():
    if not model or not scaler:
        return "Model not loaded. Please check server logs.", 500
        
    # Comprehensive feature descriptions
    FEATURE_DESCRIPTIONS = {
        # Numerical features
        "lead_time": "Number of days between booking date and arrival date",
        "stays_in_weekend_nights": "Number of weekend nights (Saturday or Sunday) the guest stayed",
        "stays_in_week_nights": "Number of week nights (Monday to Friday) the guest stayed",
        "adults": "Number of adults",
        "children": "Number of children",
        "babies": "Number of babies",
        "previous_cancellations": "Number of previous bookings canceled by the guest",
        "previous_bookings_not_canceled": "Number of previous bookings not canceled by the guest",
        "booking_changes": "Number of changes/amendments made to the booking",
        "days_in_waiting_list": "Number of days the booking was on the waiting list",
        "adr": "Average Daily Rate (average rental income per paid occupied room)",
        "required_car_parking_spaces": "Number of car parking spaces required by the guest",
        "total_of_special_requests": "Number of special requests made by the guest",
        
        # Dummy variables
        "dummy_Resort Hotel": "1 if booking is for Resort Hotel, 0 for City Hotel",
        "dummy_2016": "1 if arrival year is 2016, 0 otherwise",
        "dummy_2017": "1 if arrival year is 2017, 0 otherwise",
        "dummy_1": "1 if is_repeated_guest, 0 otherwise",
        "dummy_FB": "1 if meal type is Full Board, 0 otherwise",
        "dummy_HB": "1 if meal type is Half Board, 0 otherwise",
        "dummy_SC": "1 if meal type is Self Catering, 0 otherwise",
        "dummy_Complementary": "1 if complementary meal, 0 otherwise",
        "dummy_Corporate": "1 if market segment is Corporate, 0 otherwise",
        "dummy_Direct": "1 if market segment is Direct, 0 otherwise",
        "dummy_Groups": "1 if market segment is Groups, 0 otherwise",
        "dummy_Offline TA/TO": "1 if booked via Offline Travel Agent/Tour Operator",
        "dummy_Online TA": "1 if booked via Online Travel Agent",
        "dummy_GDS": "1 if booked via Global Distribution System",
        "dummy_TA/TO": "1 if booked via Travel Agent/Tour Operator (general)",
        "dummy_Undefined": "1 if market segment is undefined",
        "dummy_B": "1 if reserved room type B, 0 otherwise",
        "dummy_C": "1 if reserved room type C, 0 otherwise",
        "dummy_D": "1 if reserved room type D, 0 otherwise",
        "dummy_E": "1 if reserved room type E, 0 otherwise",
        "dummy_F": "1 if reserved room type F, 0 otherwise",
        "dummy_G": "1 if reserved room type G, 0 otherwise",
        "dummy_H": "1 if reserved room type H, 0 otherwise",
        "dummy_L": "1 if reserved room type L, 0 otherwise",
        "dummy_P": "1 if reserved room type P, 0 otherwise",
        "dummy_Non Refund": "1 if deposit type is Non Refundable",
        "dummy_Refundable": "1 if deposit type is Refundable",
        "dummy_Group": "1 if customer type is Group",
        "dummy_Transient": "1 if customer type is Transient",
        "dummy_Transient-Party": "1 if customer type is Transient-Party",
        "dummy_Spring": "1 if arrival season is Spring (March-May)",
        "dummy_Summer": "1 if arrival season is Summer (June-August)",
        "dummy_Winter": "1 if arrival season is Winter (December-February)"
    }
    
    # Format feature names for display
    def format_feature_name(feature):
        name = feature.replace('dummy_', '').replace('_', ' ')
        
        # Special formatting rules
        if 'TA/TO' in name:
            name = name.replace('TA/TO', 'TA/TO')
        if 'adr' in name:
            name = 'Average Daily Rate (ADR)'
        if 'Non Refund' in name:
            name = 'Non-Refundable Deposit'
            
        return name.title()
    
    # Split features into logical groups
    numerical_features = FEATURE_COLUMNS[:13]
    categorical_features = FEATURE_COLUMNS[13:]
    
    # Group categorical features
    hotel_features = [f for f in categorical_features if 'Hotel' in f]
    year_features = [f for f in categorical_features if '20' in f]
    meal_features = [f for f in categorical_features if any(x in f for x in ['FB', 'HB', 'SC', 'Complementary'])]
    market_segment_features = [f for f in categorical_features if any(x in f for x in ['Corporate', 'Direct', 'Groups', 'TA/TO', 'GDS'])]
    room_type_features = [f for f in categorical_features if f.startswith('dummy_') and len(f.split('_')[-1]) == 1]
    deposit_features = [f for f in categorical_features if any(x in f for x in ['Refund', 'Non'])]
    customer_type_features = [f for f in categorical_features if any(x in f for x in ['Group', 'Transient'])]
    season_features = [f for f in categorical_features if any(x in f for x in ['Spring', 'Summer', 'Winter'])]
    
    # Other categorical features
    other_features = list(set(categorical_features) - set(
        hotel_features + year_features + meal_features + market_segment_features + 
        room_type_features + deposit_features + customer_type_features + season_features
    ))
    
    return render_template(
        "index.html",
        numerical_features=numerical_features,
        feature_descriptions=FEATURE_DESCRIPTIONS,
        format_feature_name=format_feature_name,
        
        # Grouped categorical features
        hotel_features=hotel_features,
        year_features=year_features,
        meal_features=meal_features,
        market_segment_features=market_segment_features,
        room_type_features=room_type_features,
        deposit_features=deposit_features,
        customer_type_features=customer_type_features,
        season_features=season_features,
        other_features=other_features
    )

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if not model or not scaler:
        return "Model not loaded. Please try again later.", 500
        
    try:
        # Get form data
        form_data = {}
        for feature in FEATURE_COLUMNS:
            value = request.form.get(feature, "0")
            try:
                form_data[feature] = float(value)
            except ValueError:
                form_data[feature] = 0.0
        
        # Create DataFrame with correct feature order
        df = pd.DataFrame([form_data])[FEATURE_COLUMNS]
        
        # Scale features
        X_scaled = scaler.transform(df)
        
        # Make prediction
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0][1]
        
        # Format output
        prediction_text = "WILL be canceled" if pred == 1 else "will NOT be canceled"
        confidence = proba if pred == 1 else 1 - proba
        
        return render_template(
            "result.html",
            prediction=pred,
            prediction_text=prediction_text,
            confidence=f"{confidence:.1%}",
            features=form_data
        )
    
    except Exception as e:
        return f"Prediction error: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)