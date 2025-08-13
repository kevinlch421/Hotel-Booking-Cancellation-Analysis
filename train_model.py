# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
import pickle

# 1. Load raw data
df = pd.read_csv("/Users/kevinleungch421/Desktop/Hotel-Booking-Cancellation-Analysis/data/hotel_bookings.csv")

# 2. Random sample for efficiency
df = df.sample(n=40000, random_state=42)

# 3. Map months → seasons
month_to_season = {
    'January':'Winter','February':'Winter','March':'Spring','April':'Spring',
    'May':'Spring','June':'Summer','July':'Summer','August':'Summer',
    'September':'Autumn','October':'Autumn','November':'Autumn','December':'Winter'
}
df['season'] = df['arrival_date_month'].map(month_to_season)

# 4. Drop useless columns
drop_cols = [
    'assigned_room_type','country','arrival_date_week_number',
    'arrival_date_day_of_month','agent','company',
    'reservation_status_date','reservation_status','arrival_date_month'
]
df.drop(columns=[c for c in drop_cols if c in df], inplace=True)

# 5. Fill and cast children
df['children'].fillna(0, inplace=True)
df['children'] = df['children'].astype(int)

# 6. Merge 'Undefined' → 'SC' in meal
df['meal'].replace('Undefined', 'SC', inplace=True)

# 7. Cap outliers at ±1 std
numeric_cols = [
    'lead_time','stays_in_weekend_nights','stays_in_week_nights',
    'adr','total_of_special_requests'
]
for c in numeric_cols:
    μ, σ = df[c].mean(), df[c].std()
    df[c] = np.clip(df[c], μ - σ, μ + σ)

# 8. One-hot encode (drop_first=True)
to_encode = [
    'hotel','arrival_date_year','is_repeated_guest','meal',
    'market_segment','distribution_channel','reserved_room_type',
    'deposit_type','customer_type','season'
]
df_encoded = pd.get_dummies(df, columns=to_encode, prefix='dummy', drop_first=True)

# 9. Split X / y
y = df_encoded.pop('is_canceled')
X = df_encoded

# 10. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# 11. Fit scaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 12. Save scaler
with open("minmax_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# 13. Train model
gb = GradientBoostingClassifier(n_estimators=500, max_depth=7, random_state=1)
gb.fit(X_train_scaled, y_train)

# 14. Save model
with open("gradient_boosting_hotel_cancellation.pkl", "wb") as f:
    pickle.dump(gb, f)

print("Training complete. Saved minmax_scaler.pkl & gradient_boosting_hotel_cancellation.pkl")