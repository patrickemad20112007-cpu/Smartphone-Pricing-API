import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib 







np.random.seed(42) # For reproducibility

data = {
    'Brand': np.random.choice(['Samsung', 'Apple', 'Xiaomi', 'Oppo', 'Vivo'], 1000),
    'RAM_GB': np.random.choice([4, 6, 8, 12, 16], 1000),
    'Storage_GB': np.random.choice([64, 128, 256, 512, 1024], 1000),
    'Battery_mAh': np.random.randint(3000, 6000, 1000),
    'Camera_MP': np.random.choice([12, 48, 50, 64, 108], 1000),
    # Introducing some missing values intentionally for you to handle later!
    'Screen_Size_inch': np.random.choice([6.1, 6.4, 6.7, 6.8, np.nan], 1000, p=[0.2, 0.3, 0.3, 0.15, 0.05])
}

df = pd.DataFrame(data)

# Logic to calculate a realistic base price
base_price = (df['RAM_GB'] * 15) + (df['Storage_GB'] * 0.8) + (df['Camera_MP'] * 2)
brand_multiplier = df['Brand'].map({'Apple': 1.5, 'Samsung': 1.3, 'Xiaomi': 0.9, 'Oppo': 0.95, 'Vivo': 0.95})
df['Price'] = (base_price * brand_multiplier) + np.random.normal(0, 50, 1000)

df.to_csv('smartphones_pricing.csv', index=False)



df=pd.read_csv("smartphones_pricing.csv")
X = df.drop(columns=['Price'])
Y = df["Price"]

X_train,X_test,y_train,y_test=train_test_split(X , Y , test_size=.2, random_state=42)

print("Total data size:",df.shape)
print("Training data size:", X_train.shape)
print("Testing data size:", X_test.shape)


num_features=['RAM_GB', 'Storage_GB', 'Battery_mAh', 'Camera_MP', 'Screen_Size_inch']
cat_features=["Brand"]

num_transformer=SimpleImputer(strategy="median")

cat_transformer=OneHotEncoder(handle_unknown='ignore')

preprocessor=ColumnTransformer(transformers=[('num',num_transformer,num_features),('cat',cat_transformer,cat_features)])

X_train_processed=preprocessor.fit_transform(X_train)
print(X_train_processed.shape)

my_pipeline=Pipeline(steps=[("processor",preprocessor),("model",RandomForestRegressor(random_state=42))])
my_pipeline.fit(X_train,y_train)




y_pred = my_pipeline.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error : ${mae:.2f}")
print(f"R2 Score : {r2:.2f}")
import pandas as pd


new_mobile = {
    'Brand': ['Xiaomi'],        
    'RAM_GB': [8],             
    'Storage_GB': [128],       
    'Battery_mAh': [4323],      
    'Camera_MP': [12],           
    'Screen_Size_inch': [6.7]   
}


new_mobile_df = pd.DataFrame(new_mobile)



predicted_price = my_pipeline.predict(new_mobile_df)

print(f"The Expected Price: ${predicted_price[0]:.2f}")


joblib.dump(my_pipeline,'smartphone_pricing_model.pkl')