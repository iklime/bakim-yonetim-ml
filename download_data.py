from ucimlrepo import fetch_ucirepo
import pandas as pd

dataset = fetch_ucirepo(id=601)

X = dataset.data.features
y = dataset.data.targets

df = pd.concat([X, y], axis=1)
df.to_csv("ai4i2020.csv", index=False)

print("Veri seti basariyla indirildi ve kaydedildi: ai4i2020.csv")
print(f"Satir sayisi: {len(df)}")
print(f"Sutunlar: {list(df.columns)}")
print(df.head())
