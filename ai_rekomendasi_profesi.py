import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Dataset lebih besar & lebih beragam
X_train = np.array([
    [8, 7, 9],  # Software Engineer
    [4, 6, 5],  # Designer
    [7, 8, 6],  # Data Scientist
    [9, 9, 9],  # Dokter
    [3, 4, 2],  # Atlet
    [5, 6, 7],  # Guru
    [6, 5, 8],  # Peneliti
    [2, 9, 5],  # Seniman
    [7, 5, 6],  # Pengusaha
    [8, 6, 9],  # Insinyur
])

y_train = np.array([
    "Software Engineer", "Designer", "Data Scientist", "Dokter", "Atlet",
    "Guru", "Peneliti", "Seniman", "Pengusaha", "Insinyur"
])

gaji_train = np.array([10000, 5000, 12000, 15000, 7000, 4500, 11000, 4000, 13000, 14000])

# Label encoding profesi
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# Normalisasi fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# RandomForest dengan lebih banyak estimator (pohon keputusan)
clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
clf.fit(X_train_scaled, y_train_encoded)

# Linear Regression untuk prediksi gaji
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, gaji_train)

# Ambil input dari command line argument atau gunakan nilai default
minat_teknologi = int(sys.argv[1]) if len(sys.argv) > 1 else 5
minat_kreativitas = int(sys.argv[2]) if len(sys.argv) > 2 else 5
minat_sains = int(sys.argv[3]) if len(sys.argv) > 3 else 5

X_new = np.array([[minat_teknologi, minat_kreativitas, minat_sains]])
X_new_scaled = scaler.transform(X_new)

# Prediksi profesi & gaji
predicted_profession_encoded = clf.predict(X_new_scaled)[0]
predicted_profession = le.inverse_transform([predicted_profession_encoded])[0]
predicted_salary = round(max(lin_reg.predict(X_new_scaled)[0], 0))  # 🔥 Cegah nilai negatif

# Output lebih rapi
print(f"\nRekomendasi Profesi: {predicted_profession}")
print(f"Estimasi Gaji: ${predicted_salary}")
