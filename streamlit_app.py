import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# Sayfa yapılandırması
st.set_page_config(
    page_title="Atık Sınıflandırma Uygulaması",
    page_icon="♻️",
    layout="wide"
)

# Model yükleme
@st.cache_resource
def load_ml_model():
    return load_model("garbageapp/mobilenet_model.h5")

model = load_ml_model()

class_names = [
    "pil", "organik atık", "kahverengi cam", "karton",
    "giysi", "yeşil cam", "metal", "kağıt",
    "plastik", "ayakkabı", "çöp", "beyaz cam"
]

# Başlık
st.title("♻️ Atık Sınıflandırma Uygulaması")

# Dosya yükleme
uploaded_file = st.file_uploader("Bir atık görseli yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Görüntüyü göster
    image = Image.open(uploaded_file)
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)
    
    # Tahmin yap
    if st.button("Sınıflandır"):
        # Görüntüyü model için hazırla
        img = image.resize((224, 224)).convert("RGB")
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        
        # Tahmin yap
        prediction = model.predict(img_array)
        class_id = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        
        # Sonuçları göster
        st.success(f"Tahmin: {class_names[class_id]}")
        st.info(f"Güven: {confidence:.2%}")
        
        # Tüm sınıfların olasılıklarını göster
        st.subheader("Tüm Sınıfların Olasılıkları")
        for i, (class_name, prob) in enumerate(zip(class_names, prediction[0])):
            st.progress(float(prob))
            st.write(f"{class_name}: {prob:.2%}")

# Harita sayfası
st.sidebar.title("Navigasyon")
if st.sidebar.button("Harita"):
    st.subheader("Atık Toplama Noktaları")
    # Burada harita entegrasyonu yapılabilir
    st.write("Harita özelliği yakında eklenecek!") 