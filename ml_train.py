import os
import glob
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from skimage import filters
import joblib

IMAGES_DIR = "data/images/*"
MASKS_DIR = "data/masks/*"
SAMPLES_PER_IMAGE = 30000

def extract_advanced_features(image_gray):
    """Zoptymalizowana ekstrakcja cech z okna 5x5 bez zabijania pamięci RAM."""
    # 1. CLAHE (Jasność bazowa)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image_gray)
    
    # 2. Średnia lokalna w oknie 5x5
    mean_5x5 = cv2.blur(enhanced, (5, 5))
    
    # 3. Odchylenie standardowe w oknie 5x5 (Wariancja koloru)
    # std = sqrt(mean(x^2) - mean(x)^2)
    img_sq = cv2.multiply(enhanced.astype(np.float32), enhanced.astype(np.float32))
    mean_sq = cv2.blur(img_sq, (5, 5))
    sq_mean = cv2.multiply(mean_5x5.astype(np.float32), mean_5x5.astype(np.float32))
    var_5x5 = np.sqrt(np.abs(mean_sq - sq_mean)).astype(np.uint8)
    
    # 4. Krawędzie (Sobel)
    sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.normalize(np.sqrt(sobelx**2 + sobely**2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 5. Filtr Frangiego
    green_inv = cv2.bitwise_not(enhanced)
    frangi_img = filters.frangi(green_inv, sigmas=range(1, 4, 1), black_ridges=False)
    frangi_norm = cv2.normalize(frangi_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Złożenie wszystkich cech w jeden wektor (H*W, 5)
    X = np.column_stack((
        enhanced.flatten(),
        mean_5x5.flatten(),
        var_5x5.flatten(),
        sobel_mag.flatten(),
        frangi_norm.flatten()
    ))
    return X

def main():
    image_paths = sorted(glob.glob(IMAGES_DIR))
    mask_paths = sorted(glob.glob(MASKS_DIR))
    
    if not image_paths or not mask_paths:
        print("Nie znaleziono obrazów.")
        return

    print("Rozpoczynam ekstrakcję zaawansowanych cech (Feature Engineering)...")
    X_all, y_all = [], []
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        green_channel = img[:, :, 1]
        
        # Nowa ekstrakcja cech
        X_features = extract_advanced_features(green_channel)
        y_labels = (mask.flatten() > 127).astype(int)
        
        # Ograniczenie próbek
        indices = np.random.choice(len(y_labels), SAMPLES_PER_IMAGE, replace=False)
        X_all.append(X_features[indices])
        y_all.append(y_labels[indices])
        print(f"Przetworzono: {os.path.basename(img_path)}")

    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Wyrównywanie klas (Undersampling)...")
    rus = RandomUnderSampler(random_state=42)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
    
    print("Rozpoczynam trening zaawansowanego modelu Random Forest...")
    # Większa liczba drzew (100) i nieco większa głębokość poprawi skuteczność
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    clf.fit(X_train_res, y_train_res)
    
    print("Ocena na zbiorze hold-out:")
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    
    model_name = "random_forest_model.pkl"
    joblib.dump(clf, model_name)
    print(f"Zapisano model! Podmień funkcję w app.py aby go użyć.")

if __name__ == "__main__":
    main()