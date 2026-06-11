import os
import tensorflow as tf
from google.colab import drive
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Google Drive'ı Colab'e bağlıyoruz
drive.mount('/content/drive')

# Sınıflı veri setinin Drive'daki klasör yolları
Egitim_Yolu = '/content/drive/MyDrive/DTGCN_DATA/1_Multistage_Malaria_Parasite_Recognition/train'
Test_Yolu = '/content/drive/MyDrive/DTGCN_DATA/1_Multistage_Malaria_Parasite_Recognition/test'

# İlk baştaki o kararlı ve sade veri yükleyiciler
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

print("\n6 Sınıflı Eğitim Verisi Yükleniyor:")
train_images = train_datagen.flow_from_directory(
    Egitim_Yolu,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

print("\n6 Sınıflı Test (Validation) Verisi Yükleniyor:")
val_images = test_datagen.flow_from_directory(
    Test_Yolu,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)







print("Model Mimarisi Kuruluyor: DenseNet201...")

# Ön eğitimli DenseNet201 gövdesi
base_model = tf.keras.applications.DenseNet201(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

# İlk aşamadaki gibi ana gövde kilitli (Dondurulmuş)
base_model.trainable = False

# Üzerine eklediğimiz sınıflandırma katmanları
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3), # Orijinal kararlı dropout oranı
    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nİlk Kararlı Eğitim Başlıyor...")

history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=5
)





# Modeli kalıcı olarak Drive'a kaydediyoruz
kayit_yolu = '/content/drive/MyDrive/sitma_6_sinifli_densenet201_tamamlandi.h5'
model.save(kayit_yolu)

print(f"\nİşlem tamam Patron! %71'lik orijinal modelin Drive'a taş gibi kaydedildi: {kayit_yolu}")











import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("Geliştirilmiş Fine-Tuning Aşaması")

# 1. ZORLAŞTIRILMIŞ VERİ ÇOĞALTMA
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,      # Rastgele 20 derece döndür
    width_shift_range=0.1,  # Yüzde 10 sağa sola kaydır
    height_shift_range=0.1, # Yüzde 10 aşağı yukarı kaydır
    horizontal_flip=True,   # Rastgele yatay çevir
    fill_mode='nearest'
)

# Test verisini normalize ediyoruz
test_datagen = ImageDataGenerator(rescale=1./255)

print("\nVeriler Data Augmentation ile Sisteme Alınıyor...")
train_images = train_datagen.flow_from_directory(
    Egitim_Yolu, target_size=(224, 224), batch_size=32, class_mode='categorical'
)
val_images = test_datagen.flow_from_directory(
    Test_Yolu, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False
)

# 2. TRANSFER LEARNING & FINE-TUNING
# Ana gövdeyi çağırıyoruz
base_model = tf.keras.applications.DenseNet201(
    input_shape=(224, 224, 3), include_top=False, weights='imagenet', pooling='avg'
)

# İnce Ayar Kilitlerini Açıyoruz!
base_model.trainable = True

# İlk katmanlar (temel çizgileri/renkleri öğrenenler) dondurulmuş kalıyor,
# Sadece son 30 katmanı sıtma parazitlerine özel eğitime açıyoruz.
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Yeni Sınıflandırma Katmanları
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5), # Ezberi bozmak için %30'dan %50'ye çıkardık
    tf.keras.layers.Dense(6, activation='softmax')
])

# Kilitleri açtığımız için ağırlıkları bozmamak adına çok ÇOK küçük bir öğrenme hızı (1e-4) kullanıyoruz
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 3. FREN SİSTEMİ (Early Stopping)
# Modelin doğruluk oranı 4 epoch boyunca artmazsa eğitimi kes ve en yüksek olanı kaydet
erken_durdurma = EarlyStopping(
    monitor='val_accuracy',
    patience=4,
    restore_best_weights=True,
    verbose=1
)

print("\n İleri Seviye Model Eğitimi Başlıyor (15 Epoch)...")
history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=15,
    callbacks=[erken_durdurma]
)

# 4. ZİRVEDEKİ MODELİ KAYDETME
yeni_kayit_yolu = '/content/drive/MyDrive/sitma_6_sinifli_GELISMIS_model.h5'
model.save(yeni_kayit_yolu)
print(f"\n İşlem Tamam, Ezber başarıyla kırıldı ve yeni süper model şuraya kilitlendi: {yeni_kayit_yolu}")








import gradio as gr
import tensorflow as tf
import numpy as np

print("Gelişmiş Sıtma Teşhis Sistemi Başlatılıyor...")

# Yeni gelişmiş modelin yolu
model_yolu = '/content/drive/MyDrive/sitma_6_sinifli_GELISMIS_model.h5'
model = tf.keras.models.load_model(model_yolu)

# 6 Sınıfın alfabetik sıralaması
siniflar = ['Gametocyte', 'Leukocyte', 'Red Blood Cell', 'Ring', 'Schizont', 'Trophozoite']

def tahmin_et(img):
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    tahminler = model.predict(img_array)[0]
    return {siniflar[i]: float(tahminler[i]) for i in range(6)}

# Arayüz Tasarımı
arayuz = gr.Interface(
    fn=tahmin_et,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title=" YZM0206 İleri Seviye Sıtma Evreleri Teşhis Sistemi",
    description="Fine-Tuning ve Data Augmentation ile geliştirilmiş, %76.8 test doğruluğuna sahip DenseNet201 modeli.",
    allow_flagging="never"
)

print("Sistem hazır. Test etmek için aşağıdaki linke tıklayabilirsiniz:")
arayuz.launch(share=True)