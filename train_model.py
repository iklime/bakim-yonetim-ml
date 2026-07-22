import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

# Veriyi yukle
df = pd.read_csv("ai4i2020.csv")

# Type sutununu sayisal hale getir (L/M/H -> 0/1/2)
le = LabelEncoder()
df["Type_encoded"] = le.fit_transform(df["Type"])

# Ozellikler (X) - sadece gercek sensor verileri, sizinti kolonlari HARIC
features = ["Type_encoded", "Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]
X = df[features]
y = df["Machine failure"]

# Egitim/test ayirma (stratify ile sinif dengesini koruyoruz, cunku arizalar nadir)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model egitimi (class_weight=balanced, cunku arizalar veri setinde az sayida)
model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Test verisi uzerinde tahmin
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Performans metrikleri
print("=== Model Performansi ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.3f}")
print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()
print("Detayli Rapor:")
print(classification_report(y_test, y_pred))

# Ozellik onem siralamasi
print("=== Ozellik Onem Siralamasi ===")
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print(importances)

# Modeli ve encoder'i kaydet
joblib.dump(model, "failure_model.pkl")
joblib.dump(le, "type_encoder.pkl")
print()
print("Model kaydedildi: failure_model.pkl")
