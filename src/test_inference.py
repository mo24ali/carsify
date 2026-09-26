import pandas as pd
import joblib

# 1. Chargement du pipeline complet (scaler + modèle)
model_path = "models/best_car_price_model.pkl"
pipeline = joblib.load(model_path)

# 2. Simulation d'un nouveau véhicule à prédire
# (le pipeline applique lui-même le StandardScaler et le OneHotEncoder)
new_car = pd.DataFrame([{
    'year': 2018,
    'km_driven': 65000,
    'fuel': 'Diesel',
    'seller_type': 'Individual',
    'transmission': 'Manual',
    'owner': 'First Owner'
}])

# 3. Prédiction
predicted_price = pipeline.predict(new_car)[0]

print("=== TEST DE PRÉDICTION EN CONDITIONS RÉELLES ===")
print(f"Prix estimé pour le véhicule : {predicted_price:,.2f} DH")