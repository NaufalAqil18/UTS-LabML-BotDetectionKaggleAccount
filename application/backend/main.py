import uuid
import uvicorn
import joblib
from fastapi import File
from fastapi import UploadFile
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic import EmailStr
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# mmebuat instance FastAPI
app = FastAPI()

# mendefinisikan schema input 
class UserInput(BaseModel):
    name: str
    gender: str
    email_id: str
    is_glogin: bool
    follower_count: int
    following_count: int
    dataset_count: int
    code_count: int
    discussion_count: int
    avg_nb_read_time_min: float
    total_votes_gave_nb: int
    total_votes_gave_ds: int
    total_votes_gave_dc: int


def preprocess_pipeline():
    numeric_features = ['follower_count', 'following_count', 'dataset_count', 'code_count', 
                         'discussion_count', 'avg_nb_read_time_min', 'total_votes_gave_nb', 
                         'total_votes_gave_ds', 'total_votes_gave_dc']  # masukkan kolom-kolom numerik

    categorical_boolean_features = ['gender', 'is_glogin']  # masukkan kolom kategorikal dan boolean

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())  # Standarisasi data numerik
    ])

    categorical_boolean_transfomer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Encoding untuk data kategorikal
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_boolean_transfomer, categorical_boolean_features)
        ],
        remainder='drop'  # Drop kolom yang tidak termasuk dalam transformasi
    )

    return preprocessor

# load model
model = joblib.load('../../model/ExtraTrees_model.pkl')    ## SESUAIKAN DENGAN LOKASI MODEL YANG ANDA GUNAKAN

# endpoint untuk menerima input dan menghasilkan prediksi
@app.post("/predict/", summary="Melakukan klasifikasi apakah suatu user tergolong bot atau bukan")
async def predict(user_input: UserInput):
    # Ubah input menjadi format yang sesuai (pandas DataFrame)
    data = pd.DataFrame({
        'name': [user_input.name],
        'gender': [user_input.gender],
        'email_id': [user_input.email_id],
        'is_glogin': [user_input.is_glogin],
        'follower_count': [user_input.follower_count],
        'following_count': [user_input.following_count],
        'dataset_count': [user_input.dataset_count],
        'code_count': [user_input.code_count],
        'discussion_count': [user_input.discussion_count],
        'avg_nb_read_time_min': [user_input.avg_nb_read_time_min],
        'total_votes_gave_nb': [user_input.total_votes_gave_nb],
        'total_votes_gave_ds': [user_input.total_votes_gave_ds],
        'total_votes_gave_dc': [user_input.total_votes_gave_dc]
    })

    # Terapkan pipeline untuk preprocessing
    preprocessing_pipeline = preprocess_pipeline()
    processed_data = preprocessing_pipeline.fit_transform(data) 
    
    # Prediksi dengan model yang sudah dilatih
    prediction = model.predict(processed_data)
    
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)