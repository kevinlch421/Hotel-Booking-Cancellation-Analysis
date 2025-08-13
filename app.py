from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load artifacts
with open("minmax_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("gradient_boosting_hotel_cancellation.pkl", "rb") as f:
    model = pickle.load(f)

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

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", features=FEATURE_COLUMNS)

@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    form_data = {}
    for feature in FEATURE_COLUMNS:
        value = request.form.get(feature, "0")
        try:
            form_data[feature] = float(value)
        except ValueError:
            form_data[feature] = 0.0
    
    # Create DataFrame
    df = pd.DataFrame([form_data])
    
    # Scale features
    X_scaled = scaler.transform(df)
    
    # Make prediction
    pred = model.predict(X_scaled)[0]
    prediction_text = "WILL be canceled" if pred == 1 else "will NOT be canceled"
    
    return render_template(
        "result.html",
        prediction=pred,
        prediction_text=prediction_text,
        features=form_data
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)