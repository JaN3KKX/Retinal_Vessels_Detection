import streamlit as st
import numpy as np
import cv2
from skimage import io, filters
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.metrics import geometric_mean_score
import joblib
import tensorflow as tf
import os
from pathlib import Path
from skimage.util import view_as_windows
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =============================================================
# --- MONKEY PATCH ---
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
# --- STATE MANAGEMENT ---
# =============================================================
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

# =============================================================
# --- IMAGE PROCESSING LOGIC ---
# =============================================================
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
    if len(image.shape) == 3: gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else: gray = image
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) > 0

def process_frangi(image, fov_mask):
    if len(image.shape) == 3: green = image[:, :, 1]
    else: green = image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green_clahe = clahe.apply(green)
    green_inv = cv2.bitwise_not(green_clahe)
    frangi_img = filters.frangi(green_inv, sigmas=range(1, 5, 1), black_ridges=False)
    frangi_norm = cv2.normalize(frangi_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary_mask = cv2.threshold(frangi_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_mask = clean_vessel_mask(binary_mask, min_area=30)
    
    kernel = np.ones((7,7), np.uint8)
    fov_eroded = cv2.erode(fov_mask.astype(np.uint8), kernel, iterations=2) > 0
    binary_mask[~fov_eroded] = 0
    return binary_mask

@st.cache_resource
def load_ml_model(model_path="random_forest_model.pkl"):
    try: return joblib.load(model_path)
    except Exception: return None

def extract_advanced_features(image_gray):
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

def optimize_ml_threshold(image_rgb, model, fov_mask, gt_mask):
    if len(image_rgb.shape) == 3: green_channel = image_rgb[:, :, 1]
    else: green_channel = image_rgb
    h, w = green_channel.shape
    
    X = extract_advanced_features(green_channel)
    fov_flat = fov_mask.flatten()
    
    best_gmean = 0.0
    best_t = 0.50
    
    if np.any(fov_flat) and gt_mask is not None:
        probabilities = model.predict_proba(X[fov_flat])[:, 1]
        
        kernel = np.ones((7,7), np.uint8)
        fov_eroded = cv2.erode(fov_mask.astype(np.uint8), kernel, iterations=2) > 0
        
        for t in np.arange(0.10, 0.95, 0.05):
            predictions_flat = np.zeros(h * w, dtype=np.uint8)
            predictions_flat[fov_flat] = (probabilities > t).astype(np.uint8) * 255
            
            pred_mask = predictions_flat.reshape((h, w))
            pred_mask = clean_vessel_mask(pred_mask, min_area=60)
            pred_mask[~fov_eroded] = 0
            
            y_true_flat = gt_mask[fov_mask].flatten() // 255
            y_pred_flat = pred_mask[fov_mask].flatten() // 255
            
            cm = confusion_matrix(y_true_flat, y_pred_flat)
            if cm.shape == (2, 2): 
                tn, fp, fn, tp = cm.ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                gmean = np.sqrt(sens * spec)
            else:
                gmean = 0.0
                
            if gmean > best_gmean:
                best_gmean = gmean
                best_t = t
                
    return float(best_t), best_gmean

def predict_ml(image, model, fov_mask, confidence_threshold=0.5):
    if len(image.shape) == 3: green_channel = image[:, :, 1]
    else: green_channel = image
    h, w = green_channel.shape
    
    X = extract_advanced_features(green_channel)
    fov_flat = fov_mask.flatten()
    predictions = np.zeros(h * w, dtype=np.uint8)
    
    if np.any(fov_flat):
        probabilities = model.predict_proba(X[fov_flat])[:, 1]
        predictions[fov_flat] = (probabilities > confidence_threshold).astype(np.uint8) * 255
        
    pred_mask = predictions.reshape((h, w))
    pred_mask = clean_vessel_mask(pred_mask, min_area=60)
    
    kernel = np.ones((7,7), np.uint8)
    fov_eroded = cv2.erode(fov_mask.astype(np.uint8), kernel, iterations=2) > 0
    pred_mask[~fov_eroded] = 0
    
    return pred_mask

@st.cache_resource
def load_dl_model(model_path="unet_resnet50_ultimate.keras"):
    try:
        def tversky(y_true, y_pred, smooth=1e-6): return y_pred
        def focal_tversky_loss(y_true, y_pred): return y_pred
        def jaccard_coef(y_true, y_pred): return y_pred
        def bce_focal_tversky_loss(y_true, y_pred): return y_pred
        return tf.keras.models.load_model(model_path, custom_objects={'bce_focal_tversky_loss': bce_focal_tversky_loss, 'focal_tversky_loss': focal_tversky_loss, 'tversky': tversky, 'jaccard_coef': jaccard_coef}, compile=False)
    except Exception as e:
        st.warning(f"Failed to load DL model. Check file path. Error: {str(e)}")
        return None

def predict_dl_advanced(image, model, fov_mask=None, use_tta=True, use_postprocess=True, thickness=0):
    target_size = (512, 512) 
    original_size = (image.shape[1], image.shape[0])
    
    if len(image.shape) == 3: green = image[:, :, 1]
    else: green = image
        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green_clahe = clahe.apply(green)
    img_clahe = cv2.merge([green_clahe, green_clahe, green_clahe])
        
    img_resized = cv2.resize(img_clahe, target_size)
    preprocess_input = sm.get_preprocessing('resnet50')
    img_preprocessed = preprocess_input(img_resized.astype(np.float32))
    
    if use_tta:
        img_1 = img_preprocessed
        img_2 = np.fliplr(img_1)
        img_3 = np.flipud(img_1)
        img_4 = np.rot90(img_1, 2)
        p1 = model.predict(np.expand_dims(img_1, axis=0), verbose=0)[0]
        p2 = np.fliplr(model.predict(np.expand_dims(img_2, axis=0), verbose=0)[0])
        p3 = np.flipud(model.predict(np.expand_dims(img_3, axis=0), verbose=0)[0])
        p4 = np.rot90(model.predict(np.expand_dims(img_4, axis=0), verbose=0)[0], -2)
        prediction_prob = (p1 + p2 + p3 + p4) / 4.0
    else:
        img_input = np.expand_dims(img_preprocessed, axis=0)
        prediction_prob = model.predict(img_input, verbose=0)[0]
        
    prediction_prob = np.squeeze(prediction_prob)
    prob_map_255 = (prediction_prob * 255).astype(np.uint8)
    prob_map_resized = cv2.resize(prob_map_255, original_size, interpolation=cv2.INTER_LINEAR)
    _, binary_mask_small = cv2.threshold(prob_map_255, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_mask = cv2.resize(binary_mask_small, original_size, interpolation=cv2.INTER_NEAREST)
    
    if use_postprocess: binary_mask = clean_vessel_mask(binary_mask, min_area=40)
    if thickness != 0: binary_mask = adjust_vessel_thickness(binary_mask, thickness_modifier=thickness)
        
    if fov_mask is not None:
        binary_mask[~fov_mask] = 0
        prob_map_resized[~fov_mask] = 0
        
    return binary_mask, prob_map_resized

def calculate_metrics(y_true, y_pred, fov_mask):
    y_true_flat = y_true[fov_mask].flatten() // 255
    y_pred_flat = y_pred[fov_mask].flatten() // 255
    
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    if cm.shape == (2, 2): tn, fp, fn, tp = cm.ravel()
    else: tn, fp, fn, tp = 0, 0, 0, 0
        
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f_score = f1_score(y_true_flat, y_pred_flat)
    g_mean = geometric_mean_score(y_true_flat, y_pred_flat)
    return {
        "Accuracy": accuracy, "Sensitivity": sensitivity, "Specificity": specificity, 
        "Precision": precision, "F-score": f_score, "G-mean": g_mean, 
        "TP": tp, "FP": fp, "FN": fn, "TN": tn
    }

# =============================================================
# --- STREAMLIT UI ---
# =============================================================
st.set_page_config(layout="wide", page_title="Retinal Blood Vessel Detection")

st.title("Retinal Blood Vessel Detection")
st.divider()

def load_image_from_file(file_path):
    if file_path.lower().endswith('.ppm'): return io.imread(file_path)
    else: return cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

def get_mask_path(image_name):
    image_base = Path(image_name).stem
    mask_path = f"data/masks/{image_base}.vk.ppm"
    if os.path.exists(mask_path): return mask_path
    return None

images_dir = "data/images"
available_images = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.ppm', '.png', '.jpg', '.jpeg'))]) if os.path.exists(images_dir) else []

st.subheader("1. Data Selection")
tab1, tab2 = st.tabs(["Database", "Upload Custom Files"])

image_rgb, uploaded_mask = None, None
current_gt = None 

with tab1:
    if available_images:
        selected_image = st.selectbox("Select image from database:", available_images, key="image_select", on_change=reset_results)
        if selected_image:
            image_rgb = load_image_from_file(os.path.join(images_dir, selected_image))
            mask_path = get_mask_path(selected_image)
            if mask_path and os.path.exists(mask_path):
                st.markdown("Mask automatically matched.")
                uploaded_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else: 
                st.markdown("Expert mask not found for this image.")
    else: 
        st.warning("No images found in the data/images directory.")

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

# --- SIDEBAR ---
with st.sidebar:
    method = st.radio(
        "Algorithm Selection", 
        ("Baseline (Filters)", "Machine Learning (Random Forest)", "Deep Learning (U-Net SOTA 5.0)"),
        label_visibility="collapsed",
        on_change=handle_method_change
    )
    
    use_tta, use_post, thickness_val = False, False, 0
    
    if method == "Machine Learning (Random Forest)":
        st.divider()
        
        # --- AUTOMATIC OPTIMIZATION LOGIC ---
        if not st.session_state.auto_optimized and current_gt is not None and image_rgb is not None:
            with st.spinner("Auto-optimizing threshold for best G-mean..."):
                model_ml = load_ml_model()
                if model_ml:
                    fov_mask = get_fov_mask(image_rgb)
                    best_t, best_gmean = optimize_ml_threshold(image_rgb, model_ml, fov_mask, current_gt)
                    st.session_state.ml_conf_threshold = float(best_t)
                    st.session_state.auto_optimized = True
                    st.session_state.trigger_process = True
                    st.toast(f"Optimized Threshold: {best_t:.2f} (G-mean: {best_gmean:.4f})")
                    st.rerun()

        ml_conf = st.slider(
            "Confidence Threshold", 
            min_value=0.10, max_value=0.90, step=0.01,
            key="ml_conf_threshold", 
            on_change=set_trigger
        )
        
    elif method == "Deep Learning (U-Net SOTA 5.0)":
        st.divider()
        use_tta = st.checkbox("Use Test-Time Augmentation", value=True, on_change=set_trigger)
        use_post = st.checkbox("Enable Post-processing", value=True, on_change=set_trigger)
        thickness_val = st.slider(
            "Vessel thickness adjustment", 
            min_value=-3, max_value=3, value=0, 
            on_change=set_trigger
        )

# --- PROCESSING TRIGGER ---
st.divider()
if st.button("Run Analysis", width="stretch"):
    set_trigger()

if st.session_state.trigger_process:
    if image_rgb is not None:
        fov_mask = get_fov_mask(image_rgb)
        result_mask, prob_map = None, None
        
        with st.spinner(f"Analyzing image using {method}..."):
            if method == "Baseline (Filters)":
                result_mask = process_frangi(image_rgb, fov_mask)
            elif method == "Machine Learning (Random Forest)":
                model_ml = load_ml_model()
                if model_ml: 
                    result_mask = predict_ml(image_rgb, model_ml, fov_mask, confidence_threshold=st.session_state.ml_conf_threshold)
            elif method == "Deep Learning (U-Net SOTA 5.0)":
                model_dl = load_dl_model()
                if model_dl:
                    result_mask, prob_map = predict_dl_advanced(
                        image_rgb, model_dl, fov_mask=fov_mask, 
                        use_tta=use_tta, use_postprocess=use_post, thickness=thickness_val
                    )

        metrics = None
        
        if result_mask is not None and current_gt is not None:
            metrics = calculate_metrics(current_gt, result_mask, fov_mask)

        if result_mask is not None:
            st.session_state.results = {
                "image_rgb": image_rgb.copy(),
                "result_mask": result_mask,
                "prob_map": prob_map,
                "gt_mask": current_gt,
                "metrics": metrics
            }
    st.session_state.trigger_process = False

# --- RENDER RESULTS ---
if st.session_state.results is not None:
    res = st.session_state.results
    img = res["image_rgb"]
    res_mask = res["result_mask"]
    prob = res["prob_map"]
    gt = res["gt_mask"]
    met = res["metrics"]
    
    st.subheader("2. Visualization")
    st.image(res_mask, caption="Generated Binary Mask", width="stretch")
    
    if prob is not None:
        with st.expander("Show Probability Heatmap"):
            st.image(prob, clamp=True, width="stretch")
        
    overlay = img.copy()
    overlay[res_mask > 0] = [255, 0, 0]
    
    v_col1, v_col2 = st.columns(2)
    v_col1.image(img, caption="Original Image", width="stretch")
    v_col2.image(overlay, caption="Detected Vessels Overlay", width="stretch")
    
    if gt is not None and met is not None:
        st.divider()
        st.subheader("3. Statistical Analysis")
        
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Accuracy", f"{met['Accuracy']:.4f}")
        m_col2.metric("Sensitivity", f"{met['Sensitivity']:.4f}")
        m_col3.metric("Specificity", f"{met['Specificity']:.4f}")
        m_col4.metric("F-score", f"{met['F-score']:.4f}")
        m_col5.metric("G-mean", f"{met['G-mean']:.4f}")
        
        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            st.image(gt, caption="Ground Truth Mask", width="stretch")
        with c2:
            st.image(res_mask, caption="Generated Mask", width="stretch")
            
        st.markdown("**Confusion Matrix (Relative to Ground Truth)**")
        
        total_vessels = met['TP'] + met['FN']
        total_bg = met['TN'] + met['FP']
        
        cm_data = {
            "Metric": [
                "Sensitivity (True Positive Rate)", 
                "False Negative Rate", 
                "Specificity (True Negative Rate)", 
                "False Positive Rate"
            ],
            "Definition": [
                "Actual vessels correctly detected",
                "Actual vessels missed by algorithm",
                "Background correctly ignored",
                "Background falsely detected as vessel"
            ],
            "Percentage": [
                f"{(met['TP'] / total_vessels) * 100:.2f}%" if total_vessels > 0 else "0.00%",
                f"{(met['FN'] / total_vessels) * 100:.2f}%" if total_vessels > 0 else "0.00%",
                f"{(met['TN'] / total_bg) * 100:.2f}%" if total_bg > 0 else "0.00%",
                f"{(met['FP'] / total_bg) * 100:.2f}%" if total_bg > 0 else "0.00%"
            ]
        }
        st.table(pd.DataFrame(cm_data))