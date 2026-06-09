import streamlit as st
import numpy as np
import cv2
import joblib
import tensorflow as tf
import os
import pandas as pd
from skimage import io
from pathlib import Path
from sklearn.metrics import confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from core.processing import (
    clean_vessel_mask, adjust_vessel_thickness, get_fov_mask, 
    process_baseline, extract_ml_features, predict_ml, predict_dl, calculate_metrics
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGES_DIR = os.path.join(ROOT_DIR, "data", "test_data", "test_images")
TEST_MASKS_DIR = os.path.join(ROOT_DIR, "data", "test_data", "test_masks")
MODEL_ML_PATH = os.path.join(ROOT_DIR, "models", "ml_models", "random_forest_model_v3.pkl")
MODEL_DL_PATH = os.path.join(ROOT_DIR, "models", "dl_models", "unet_resnet50_ultimate_v3.keras")

if 'trigger_process' not in st.session_state:
    st.session_state.trigger_process = False

if 'results' not in st.session_state:
    st.session_state.results = None

if 'ml_conf_threshold' not in st.session_state:
    st.session_state.ml_conf_threshold = 0.50

if 'auto_optimized' not in st.session_state:
    st.session_state.auto_optimized = False

def set_trigger():
    st.session_state.trigger_process = True

def reset_results():
    st.session_state.results = None
    st.session_state.trigger_process = False
    st.session_state.auto_optimized = False

def handle_method_change():
    st.session_state.trigger_process = True
    st.session_state.auto_optimized = False

def optimize_ml_threshold(image_rgb, model, fov_mask, gt_mask):
    green_channel = image_rgb[:, :, 1] if len(image_rgb.shape) == 3 else image_rgb
    h, w = green_channel.shape
    X = extract_ml_features(green_channel)
    fov_flat = fov_mask.flatten()
    best_gmean = 0.0
    best_t = 0.50
    
    if np.any(fov_flat) and gt_mask is not None:
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
                best_t = t
                
    return round(float(best_t), 2), best_gmean

@st.cache_resource
def load_ml_model(model_path=MODEL_ML_PATH):
    try:
        return joblib.load(model_path)
    except Exception:
        return None

@st.cache_resource
def load_dl_model(model_path=MODEL_DL_PATH):
    try:
        def tversky(y_true, y_pred, smooth=1e-6): return y_pred
        def focal_tversky_loss(y_true, y_pred): return y_pred
        def jaccard_coef(y_true, y_pred): return y_pred
        def bce_focal_tversky_loss(y_true, y_pred): return y_pred
        return tf.keras.models.load_model(model_path, custom_objects={'bce_focal_tversky_loss': bce_focal_tversky_loss, 'focal_tversky_loss': focal_tversky_loss, 'tversky': tversky, 'jaccard_coef': jaccard_coef}, compile=False)
    except Exception:
        return None

st.set_page_config(layout="wide", page_title="Retinal Blood Vessel Detection")
st.title("Retinal Blood Vessel Detection")
st.divider()

def load_image_from_file(file_path):
    if file_path.lower().endswith('.ppm'): return io.imread(file_path)
    else: return cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

def get_mask_path(image_name):
    image_base = Path(image_name).stem
    mask_path = os.path.join(TEST_MASKS_DIR, f"{image_base}.vk.ppm")
    if os.path.exists(mask_path): return mask_path
    return None

available_images = sorted([f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.ppm', '.png', '.jpg', '.jpeg'))]) if os.path.exists(TEST_IMAGES_DIR) else []

st.subheader("1. Data Selection")
tab1, tab2 = st.tabs(["Database", "Upload Custom Files"])

image_rgb, uploaded_mask = None, None
current_gt = None 

with tab1:
    if available_images:
        selected_image = st.selectbox("Select image from database:", available_images, key="image_select", on_change=reset_results)
        if selected_image:
            image_rgb = load_image_from_file(os.path.join(TEST_IMAGES_DIR, selected_image))
            mask_path = get_mask_path(selected_image)
            if mask_path and os.path.exists(mask_path):
                st.markdown("Mask automatically matched.")
                uploaded_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else: 
                st.markdown("Expert mask not found for this image.")
    else: 
        st.warning("No images found in the directory.")

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        uploaded_img = st.file_uploader("Upload retinal image", type=['png', 'jpg', 'jpeg', 'ppm'], key="u_img", on_change=reset_results)
        if uploaded_img:
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            image_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    with col2:
        uploaded_mask_file = st.file_uploader("Upload expert mask", type=['png', 'jpg', 'jpeg', 'tif', 'ppm'], key="u_mask", on_change=reset_results)
        if uploaded_mask_file:
            mask_bytes = np.asarray(bytearray(uploaded_mask_file.read()), dtype=np.uint8)
            uploaded_mask = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)

if uploaded_mask is not None:
    current_gt = uploaded_mask if isinstance(uploaded_mask, np.ndarray) else cv2.imdecode(np.asarray(bytearray(uploaded_mask.read()), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if len(np.unique(current_gt)) > 2 or (np.max(current_gt) > 1 and np.max(current_gt) < 255):
        _, current_gt = cv2.threshold(current_gt, 127, 255, cv2.THRESH_BINARY)

with st.sidebar:
    method = st.radio(
        "Algorithm Selection", 
        ("Baseline (Filters)", "Machine Learning (Random Forest)", "Deep Learning (U-Net)"),
        label_visibility="collapsed",
        on_change=handle_method_change
    )
    
    use_tta, use_post, thickness_val = False, False, 0
    
    if method == "Machine Learning (Random Forest)":
        st.divider()
        
        if not st.session_state.auto_optimized and current_gt is not None and image_rgb is not None:
            with st.spinner("Auto-optimizing threshold..."):
                model_ml = load_ml_model()
                if model_ml:
                    fov_mask = get_fov_mask(image_rgb)
                    best_t, best_gmean = optimize_ml_threshold(image_rgb, model_ml, fov_mask, current_gt)
                    st.session_state.ml_conf_threshold = float(best_t)
                    st.session_state.auto_optimized = True
                    st.session_state.trigger_process = True
                    st.rerun()

        ml_conf = st.slider(
            "Confidence Threshold", 
            min_value=0.10, max_value=0.90, step=0.01,
            key="ml_conf_threshold", 
            on_change=set_trigger
        )
        
    elif method == "Deep Learning (U-Net)":
        st.divider()
        use_tta = st.checkbox("Use Test-Time Augmentation", value=True, on_change=set_trigger)
        use_post = st.checkbox("Enable Post-processing", value=True, on_change=set_trigger)
        thickness_val = st.slider(
            "Vessel thickness adjustment", 
            min_value=-3, max_value=3, value=0, 
            on_change=set_trigger
        )

st.divider()
if st.button("Run Analysis", width="stretch"):
    set_trigger()

if st.session_state.trigger_process:
    if image_rgb is not None:
        fov_mask = get_fov_mask(image_rgb)
        result_mask = None
        
        with st.spinner(f"Analyzing image..."):
            if method == "Baseline (Filters)":
                result_mask = process_baseline(image_rgb, fov_mask)
            elif method == "Machine Learning (Random Forest)":
                model_ml = load_ml_model()
                if model_ml: 
                    result_mask = predict_ml(image_rgb, model_ml, fov_mask, confidence_threshold=st.session_state.ml_conf_threshold)
            elif method == "Deep Learning (U-Net)":
                model_dl = load_dl_model()
                if model_dl:
                    result_mask = predict_dl(image_rgb, model_dl, fov_mask, use_tta=use_tta, use_postprocess=use_post, thickness=thickness_val)

        metrics = None
        if result_mask is not None and current_gt is not None:
            metrics = calculate_metrics(current_gt, result_mask, fov_mask)

        if result_mask is not None:
            st.session_state.results = {
                "image_rgb": image_rgb.copy(),
                "result_mask": result_mask,
                "gt_mask": current_gt,
                "metrics": metrics,
                "fov_mask": fov_mask
            }
    st.session_state.trigger_process = False

if st.session_state.results is not None:
    res = st.session_state.results
    img = res["image_rgb"]
    res_mask = res["result_mask"]
    gt = res["gt_mask"]
    met = res["metrics"]
    fov = res["fov_mask"]
    
    st.subheader("2. Visualization")
    st.image(res_mask, caption="Generated Binary Mask", width="stretch")
    
    overlay = img.copy()
    overlay[res_mask > 0] = [0, 255, 0]
    
    v_col1, v_col2 = st.columns(2)
    v_col1.image(img, caption="Original Image", width="stretch")
    v_col2.image(overlay, caption="Detected Vessels Overlay", width="stretch")
    
    if gt is not None and met is not None:
        st.divider()
        st.subheader("3. Statistical Analysis")
        
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Accuracy", f"{met['Acc']:.4f}")
        m_col2.metric("Sensitivity", f"{met['Sens']:.4f}")
        m_col3.metric("Specificity", f"{met['Spec']:.4f}")
        m_col4.metric("F-score", f"{met['F1']:.4f}")
        m_col5.metric("G-mean", f"{met['Gm']:.4f}")
        
        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            st.image(gt, caption="Ground Truth Mask", width="stretch")
        with c2:
            st.image(res_mask, caption="Generated Mask", width="stretch")
            
        st.markdown("**Confusion Matrix (Relative to Ground Truth)**")
        
        y_true_flat = gt[fov].flatten() // 255
        y_pred_flat = res_mask[fov].flatten() // 255
        cm = confusion_matrix(y_true_flat, y_pred_flat)
        if cm.shape == (2, 2): tn, fp, fn, tp = cm.ravel()
        else: tn, fp, fn, tp = 0, 0, 0, 0
        
        total_vessels = tp + fn
        total_bg = tn + fp
        
        cm_data = {
            "Metric": [
                "Sensitivity (True Positive Rate)", 
                "False Negative Rate", 
                "Specificity (True Negative Rate)", 
                "False Positive Rate"
            ],
            "Percentage": [
                f"{(tp / total_vessels) * 100:.2f}%" if total_vessels > 0 else "0.00%",
                f"{(fn / total_vessels) * 100:.2f}%" if total_vessels > 0 else "0.00%",
                f"{(tn / total_bg) * 100:.2f}%" if total_bg > 0 else "0.00%",
                f"{(fp / total_bg) * 100:.2f}%" if total_bg > 0 else "0.00%"
            ]
        }
        st.table(pd.DataFrame(cm_data))