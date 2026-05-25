import os
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =============================================================
# --- ŁATKA (MONKEY PATCH) ---
# =============================================================
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow.keras.utils as utils
if not hasattr(utils, 'generic_utils'):
    utils.generic_utils = utils
    if not hasattr(utils.generic_utils, 'get_custom_objects'):
        import keras.saving
        utils.generic_utils.get_custom_objects = keras.saving.get_custom_objects
        
import segmentation_models as sm 
# =============================================================

IMAGES_DIR = "data/images/*"
MASKS_DIR = "data/masks/*"
TARGET_SIZE = (512, 512)  
EPOCHS = 60           
BATCH_SIZE = 2            
BACKBONE = 'resnet50'

def tversky(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)

def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)

def jaccard_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def load_data():
    image_paths = sorted(glob.glob(IMAGES_DIR))
    mask_paths = sorted(glob.glob(MASKS_DIR))
    images, masks = [], []
    preprocess_input = sm.get_preprocessing(BACKBONE)
    
    print("Wczytywanie i skalowanie obrazów (SOTA U-Net z ResNet50, 512x512)...")
    for img_path, mask_path in zip(image_paths, mask_paths):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        green = img[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        green_clahe = clahe.apply(green)
        img_clahe = cv2.merge([green_clahe, green_clahe, green_clahe])
        
        img_resized = cv2.resize(img_clahe, TARGET_SIZE)
        img_preprocessed = preprocess_input(img_resized.astype(np.float32))
        images.append(img_preprocessed)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, TARGET_SIZE)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        mask = np.expand_dims(mask, axis=-1)
        masks.append(mask)
        
    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)

def main():
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Inicjalizacja U-Net z pre-trained ResNet50 backbone (fine-tuning)...")
    model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=1, activation='sigmoid')
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                  loss=focal_tversky_loss,  # NOWA POTĘŻNA FUNKCJA BŁĘDU
                  metrics=['accuracy', tversky, jaccard_coef])
    
    model_name = "unet_resnet50_ultimate.keras"
    
    callbacks = [
        EarlyStopping(patience=10, monitor='val_jaccard_coef', mode='max', restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_name, monitor='val_jaccard_coef', mode='max', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
    ]
    
    data_gen_args = dict(rotation_range=45, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.15, horizontal_flip=True, vertical_flip=True, fill_mode='constant', cval=0)
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    seed = 42
    image_generator = image_datagen.flow(X_train, batch_size=BATCH_SIZE, seed=seed)
    mask_generator = mask_datagen.flow(y_train, batch_size=BATCH_SIZE, seed=seed)
    
    def combined_generator(img_gen, msk_gen):
        while True:
            yield (next(img_gen), next(msk_gen))
            
    train_generator = combined_generator(image_generator, mask_generator)
    
    print("\nRozpoczynam OSTATECZNY trening z Augmentacją Danych i Tversky Loss...")
    model.fit(x=train_generator, steps_per_epoch=len(X_train) // BATCH_SIZE, validation_data=(X_val, y_val), epochs=EPOCHS, callbacks=callbacks)
    
if __name__ == "__main__":
    main()