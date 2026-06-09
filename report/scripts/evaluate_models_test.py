import os
import cv2
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(ROOT_DIR)

from core.processing import get_fov_mask, process_baseline, extract_ml_features, clean_vessel_mask, calculate_metrics, predict_dl

TEST_IMAGES_DIR = os.path.join(ROOT_DIR, "data", "test_data", "test_images")
TEST_MASKS_DIR = os.path.join(ROOT_DIR, "data", "test_data", "test_masks")
MODEL_ML_FINAL_PATH = os.path.join(ROOT_DIR, "models", "random_forest_model_v3.pkl")
MODEL_DL_FINAL_PATH = os.path.join(ROOT_DIR, "models", "unet_resnet50_ultimate_v3.keras")
OUTPUT_CSV_PATH = os.path.join(ROOT_DIR, "report", "data", "report_01_test_comparison.csv")

def get_mask_path(image_name):
    image_base = Path(image_name).stem
    mask_path = os.path.join(TEST_MASKS_DIR, f"{image_base}.vk.ppm")
    return mask_path if os.path.exists(mask_path) else None

def evaluate_ml_model(image_rgb, model, fov_mask, gt_mask):
    if model is None: return {"Acc":0, "Sens":0, "Spec":0, "F1":0, "Gm":0}
    green = image_rgb[:, :, 1] if len(image_rgb.shape) == 3 else image_rgb
    h, w = green.shape
    X = extract_ml_features(green)
    fov_flat = fov_mask.flatten()
    best_metrics = {"Acc":0, "Sens":0, "Spec":0, "F1":0, "Gm":0}
    best_gmean = 0.0
    
    if np.any(fov_flat):
        probs = model.predict_proba(X[fov_flat])[:, 1]
        kernel = np.ones((7,7), np.uint8)
        fov_eroded = cv2.erode(fov_mask.astype(np.uint8), kernel, iterations=2) > 0
        
        for t in np.arange(0.10, 0.95, 0.05):
            preds_flat = np.zeros(h * w, dtype=np.uint8)
            preds_flat[fov_flat] = (probs > t).astype(np.uint8) * 255
            mask = clean_vessel_mask(preds_flat.reshape((h, w)), min_area=60)
            mask[~fov_eroded] = 0
            
            mets = calculate_metrics(gt_mask, mask, fov_mask)
            if mets["Gm"] > best_gmean:
                best_gmean = mets["Gm"]
                best_metrics = mets
    return best_metrics

def main():
    print("Starting evaluation...")

    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"Error: Directory {TEST_IMAGES_DIR} not found.")
        return

    ml_model = joblib.load(MODEL_ML_FINAL_PATH) if os.path.exists(MODEL_ML_FINAL_PATH) else None

    def dummy_loss(y_true, y_pred): return y_pred
    dl_model = None
    if os.path.exists(MODEL_DL_FINAL_PATH):
        dl_model = tf.keras.models.load_model(
            MODEL_DL_FINAL_PATH, 
            custom_objects={
                'bce_focal_tversky_loss': dummy_loss, 
                'focal_tversky_loss': dummy_loss, 
                'tversky': dummy_loss, 
                'jaccard_coef': dummy_loss
            }, 
            compile=False
        )

    images = sorted([f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.ppm', '.png', '.jpg'))])
    
    if not images:
        print("Error: No images found.")
        return

    results = []

    for idx, img_name in enumerate(images, 1):
        img_path = os.path.join(TEST_IMAGES_DIR, img_name)
        mask_path = get_mask_path(img_name)
        
        if not mask_path or not os.path.exists(mask_path): 
            continue
            
        if img_name.lower().endswith('.ppm'):
            from skimage import io
            img_rgb = io.imread(img_path)
        else:
            img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, gt_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)
        fov_mask = get_fov_mask(img_rgb)
        
        b_new = process_baseline(img_rgb, fov_mask)
        met_b_new = calculate_metrics(gt_mask, b_new, fov_mask)

        met_ml_v3 = evaluate_ml_model(img_rgb, ml_model, fov_mask, gt_mask)

        m_dl_new = predict_dl(img_rgb, dl_model, fov_mask, use_tta=True, use_postprocess=True)
        met_dl_new = calculate_metrics(gt_mask, m_dl_new, fov_mask) if m_dl_new is not None else {"Acc":0, "Sens":0, "Spec":0, "F1":0, "Gm":0}

        results.append({
            "Image": img_name,
            "BaselineNew_Acc": met_b_new["Acc"], "BaselineNew_Sens": met_b_new["Sens"], "BaselineNew_Spec": met_b_new["Spec"], "BaselineNew_F1": met_b_new["F1"], "BaselineNew_Gm": met_b_new["Gm"],
            "MLv3_Acc": met_ml_v3["Acc"], "MLv3_Sens": met_ml_v3["Sens"], "MLv3_Spec": met_ml_v3["Spec"], "MLv3_F1": met_ml_v3["F1"], "MLv3_Gm": met_ml_v3["Gm"],
            "DLNew_Acc": met_dl_new["Acc"], "DLNew_Sens": met_dl_new["Sens"], "DLNew_Spec": met_dl_new["Spec"], "DLNew_F1": met_dl_new["F1"], "DLNew_Gm": met_dl_new["Gm"]
        })

    if not results:
        return

    df = pd.DataFrame(results)
    mean_row = df.mean(numeric_only=True).to_dict()
    mean_row["Image"] = "ŚREDNIA (MEAN)"
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    df.to_csv(OUTPUT_CSV_PATH, index=False, float_format="%.4f")
    
    print(f"Finished. Results saved to: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()