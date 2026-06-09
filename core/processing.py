import os
import cv2
import numpy as np
from skimage import filters
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from imblearn.metrics import geometric_mean_score
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow.keras.utils as utils
if not hasattr(utils, 'generic_utils'):
    utils.generic_utils = utils
    if not hasattr(utils.generic_utils, 'get_custom_objects'):
        import keras.saving
        utils.generic_utils.get_custom_objects = keras.saving.get_custom_objects
import segmentation_models as sm

def clean_vessel_mask(binary_mask, min_area=40):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)
    cleaned = np.zeros_like(closed_mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned

def adjust_vessel_thickness(binary_mask, thickness_modifier=0):
    if thickness_modifier == 0: return binary_mask
    kernel_size = abs(thickness_modifier) * 2 + 1 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if thickness_modifier > 0: return cv2.dilate(binary_mask, kernel, iterations=1)
    else: return cv2.erode(binary_mask, kernel, iterations=1)

def get_fov_mask(image, threshold=10):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) > 0

def process_baseline(image, fov_mask):
    green = image[:, :, 1] if len(image.shape) == 3 else image
    green_blurred = cv2.medianBlur(green, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green_clahe = clahe.apply(green_blurred)
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    black_hat = cv2.morphologyEx(green_clahe, cv2.MORPH_BLACKHAT, kernel_bh)
    enhanced_vessels = cv2.add(cv2.bitwise_not(green_clahe), black_hat)
    frangi_img = filters.frangi(enhanced_vessels, sigmas=np.arange(0.5, 4.0, 0.5), black_ridges=False)
    frangi_norm = cv2.normalize(frangi_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    otsu_thresh, _ = cv2.threshold(frangi_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adjusted_thresh = max(10, int(otsu_thresh * 0.80))
    _, binary_mask = cv2.threshold(frangi_norm, adjusted_thresh, 255, cv2.THRESH_BINARY)
    binary_mask = clean_vessel_mask(binary_mask, min_area=30)
    kernel_fov = np.ones((9, 9), np.uint8)
    fov_eroded = cv2.erode(fov_mask.astype(np.uint8), kernel_fov, iterations=2) > 0
    binary_mask[~fov_eroded] = 0
    return binary_mask

def extract_ml_features(image_gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image_gray)
    
    mean_5x5 = cv2.blur(enhanced, (5, 5))
    img_sq = cv2.multiply(enhanced.astype(np.float32), enhanced.astype(np.float32))
    mean_sq = cv2.blur(img_sq, (5, 5))
    sq_mean = cv2.multiply(mean_5x5.astype(np.float32), mean_5x5.astype(np.float32))
    var_5x5 = np.sqrt(np.abs(mean_sq - sq_mean)).astype(np.uint8)
    
    sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.normalize(np.sqrt(sobelx**2 + sobely**2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    green_inv = cv2.bitwise_not(enhanced)
    frangi_img = filters.frangi(green_inv, sigmas=range(1, 4, 1), black_ridges=False)
    frangi_norm = cv2.normalize(frangi_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return np.column_stack((enhanced.flatten(), mean_5x5.flatten(), var_5x5.flatten(), sobel_mag.flatten(), frangi_norm.flatten()))

def predict_ml(image, model, fov_mask, confidence_threshold=0.5):
    if model is None: return None
    green = image[:, :, 1] if len(image.shape) == 3 else image
    h, w = green.shape
    X = extract_ml_features(green)
    fov_flat = fov_mask.flatten()
    preds_flat = np.zeros(h * w, dtype=np.uint8)
    
    if np.any(fov_flat):
        probs = model.predict_proba(X[fov_flat])[:, 1]
        preds_flat[fov_flat] = (probs > confidence_threshold).astype(np.uint8) * 255
        
    mask = clean_vessel_mask(preds_flat.reshape((h, w)), min_area=60)
    kernel = np.ones((7,7), np.uint8)
    fov_eroded = cv2.erode(fov_mask.astype(np.uint8), kernel, iterations=2) > 0
    mask[~fov_eroded] = 0
    return mask

def predict_dl(image, model, fov_mask, use_tta=True, use_postprocess=True, thickness=0):
    if model is None: return None
    target_size = (512, 512) 
    original_size = (image.shape[1], image.shape[0])
    green = image[:, :, 1] if len(image.shape) == 3 else image
    
    green_blurred = cv2.medianBlur(green, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green_clahe = clahe.apply(green_blurred)
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    enhanced_vessels = cv2.add(cv2.bitwise_not(green_clahe), cv2.morphologyEx(green_clahe, cv2.MORPH_BLACKHAT, kernel_bh))
    
    img_3ch = cv2.merge([enhanced_vessels, enhanced_vessels, enhanced_vessels])
    img_resized = cv2.resize(img_3ch, target_size)
    img_preprocessed = sm.get_preprocessing('resnet50')(img_resized.astype(np.float32))
    
    if use_tta:
        p1 = model.predict(np.expand_dims(img_preprocessed, axis=0), verbose=0)[0]
        p2 = np.fliplr(model.predict(np.expand_dims(np.fliplr(img_preprocessed), axis=0), verbose=0)[0])
        p3 = np.flipud(model.predict(np.expand_dims(np.flipud(img_preprocessed), axis=0), verbose=0)[0])
        p4 = np.rot90(model.predict(np.expand_dims(np.rot90(img_preprocessed, 2), axis=0), verbose=0)[0], -2)
        prediction_prob = (p1 + p2 + p3 + p4) / 4.0
    else:
        prediction_prob = model.predict(np.expand_dims(img_preprocessed, axis=0), verbose=0)[0]
        
    prob_map_255 = (np.squeeze(prediction_prob) * 255).astype(np.uint8)
    _, binary_mask_small = cv2.threshold(prob_map_255, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_mask = cv2.resize(binary_mask_small, original_size, interpolation=cv2.INTER_NEAREST)
    
    if use_postprocess: binary_mask = clean_vessel_mask(binary_mask, min_area=40)
    if thickness != 0: binary_mask = adjust_vessel_thickness(binary_mask, thickness_modifier=thickness)
    binary_mask[~fov_mask] = 0
    return binary_mask

def calculate_metrics(y_true, y_pred, fov_mask):
    y_t = y_true[fov_mask].flatten() // 255
    y_p = y_pred[fov_mask].flatten() // 255
    cm = confusion_matrix(y_t, y_p)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)
    acc = accuracy_score(y_t, y_p)
    sens = tp/(tp+fn) if (tp+fn)>0 else 0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0
    f1 = f1_score(y_t, y_p)
    gm = geometric_mean_score(y_t, y_p)
    return {"Acc": acc, "Sens": sens, "Spec": spec, "F1": f1, "Gm": gm}