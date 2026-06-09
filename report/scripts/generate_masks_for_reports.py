import os
import cv2
import sys
from pathlib import Path
import joblib
import tensorflow as tf
from skimage import io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(ROOT_DIR)

from core.processing import get_fov_mask, process_baseline, predict_ml, predict_dl

TEST_IMAGES_DIR = os.path.join(ROOT_DIR, "data", "test_data", "test_images")
TEST_MASKS_DIR = os.path.join(ROOT_DIR, "data", "test_data", "test_masks")
OUTPUT_DIR = os.path.join(ROOT_DIR, "report", "figures", "masks")

GENERAL_DIR = os.path.join(OUTPUT_DIR, "general_comparison")
ML_ABL_DIR = os.path.join(OUTPUT_DIR, "ml_ablation")
DL_ABL_DIR = os.path.join(OUTPUT_DIR, "dl_ablation")
DL_THICK_DIR = os.path.join(OUTPUT_DIR, "dl_thickness")

MODEL_ML_PATH = os.path.join(ROOT_DIR, "models", "ml_models", "random_forest_model_v3.pkl")
MODEL_DL_PATH = os.path.join(ROOT_DIR, "models", "dl_models", "unet_resnet50_ultimate_v3.keras")

for d in [GENERAL_DIR, ML_ABL_DIR, DL_ABL_DIR, DL_THICK_DIR]:
    os.makedirs(d, exist_ok=True)

def main():
    if not os.path.exists(MODEL_ML_PATH) or not os.path.exists(MODEL_DL_PATH):
        print("Error: Models not found.")
        return

    ml_model = joblib.load(MODEL_ML_PATH)
    def dummy_loss(y_true, y_pred): return y_pred
    dl_model = tf.keras.models.load_model(MODEL_DL_PATH, custom_objects={'bce_focal_tversky_loss': dummy_loss, 'focal_tversky_loss': dummy_loss, 'tversky': dummy_loss, 'jaccard_coef': dummy_loss}, compile=False)

    images = sorted([f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith(('.ppm', '.png', '.jpg'))])

    for img_name in images:
        base_name = Path(img_name).stem
        img_path = os.path.join(TEST_IMAGES_DIR, img_name)
        
        if img_name.lower().endswith('.ppm'):
            img = io.imread(img_path)
        else:
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            
        mask_path = os.path.join(TEST_MASKS_DIR, f"{base_name}.vk.ppm")
        if os.path.exists(mask_path):
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            _, gt_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)
            cv2.imwrite(os.path.join(GENERAL_DIR, f"{base_name}_gt.png"), gt_mask)
        
        fov = get_fov_mask(img)
        
        cv2.imwrite(os.path.join(GENERAL_DIR, f"{base_name}_base.png"), process_baseline(img, fov))
        cv2.imwrite(os.path.join(GENERAL_DIR, f"{base_name}_ml.png"), predict_ml(img, ml_model, fov, confidence_threshold=0.5))
        cv2.imwrite(os.path.join(GENERAL_DIR, f"{base_name}_dl.png"), predict_dl(img, dl_model, fov, use_tta=True, use_postprocess=True, thickness=0))

        if "im0163" in img_name.lower():
            thresholds = [0.1, 0.65, 0.8, 0.9]
            for th in thresholds:
                mask = predict_ml(img, ml_model, fov, confidence_threshold=th)
                cv2.imwrite(os.path.join(ML_ABL_DIR, f"{base_name}_ml_th{int(th*100)}.png"), mask)
            
            dl_combos = [
                (False, False, "notta_nopost"),
                (False, True, "notta_post"),
                (True, False, "tta_nopost"),
                (True, True, "tta_post")
            ]
            for tta, post, suffix in dl_combos:
                mask = predict_dl(img, dl_model, fov, use_tta=tta, use_postprocess=post, thickness=0)
                cv2.imwrite(os.path.join(DL_ABL_DIR, f"{base_name}_dl_{suffix}.png"), mask)
            
            thicknesses = [-2, 0, 2]
            for thk in thicknesses:
                mask = predict_dl(img, dl_model, fov, use_tta=True, use_postprocess=True, thickness=thk)
                thk_str = f"minus2" if thk == -2 else (f"plus2" if thk == 2 else "0")
                cv2.imwrite(os.path.join(DL_THICK_DIR, f"{base_name}_dl_thk_{thk_str}.png"), mask)

    print(f"Finished. Masks saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()