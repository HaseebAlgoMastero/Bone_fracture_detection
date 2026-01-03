import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
from pathlib import Path

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Bone Fracture Detection",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# Custom CSS (Professional Medical UI)
# --------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

body {
    background-color: #f8fafc;
}

/* Limit max width */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Navbar */
.navbar {
    background: linear-gradient(90deg, #020617, #0f172a);
    padding: 18px 36px;
    border-radius: 14px;
    margin-bottom: 40px;
}

.nav-title {
    font-size: 26px;
    font-weight: 800;
    color: #f8fafc;
}

.nav-subtitle {
    font-size: 14px;
    color: #cbd5e1;
}

/* Hero */
.hero {
    text-align: center;
    margin-bottom: 45px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    color: #020617;
}

.hero-subtitle {
    font-size: 18px;
    color: #475569;
    margin-top: 10px;
}

/* Cards */
.card {
    background: #ffffff;
    padding: 26px;
    border-radius: 18px;
    box-shadow: 0px 10px 30px rgba(2, 6, 23, 0.08);
}

/* Section title */
.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 14px;
    color: #020617;
}

/* Image center */
img {
    display: block;
    margin-left: auto;
    margin-right: auto;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 60px;
    font-size: 14px;
    color: #64748b;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Navbar
# --------------------------------------------------
st.markdown("""
<div class="navbar">
    <div class="nav-title">🦴 Bone Fracture Detection System</div>
    <div class="nav-subtitle">AI-powered X-ray analysis using YOLO</div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Hero Section
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">Automated Bone Fracture Detection</div>
    <div class="hero-subtitle">
        Upload an X-ray image to detect fractures with deep learning precision
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Image Resize Utility (KEY FIX)
# --------------------------------------------------
def resize_image(image, max_height=420):
    w, h = image.size
    scale = max_height / h
    new_w = int(w * scale)
    return image.resize((new_w, max_height))

# --------------------------------------------------
# Load YOLO Model (Cached)
# --------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # ensure best.pt exists

model = load_model()

# --------------------------------------------------
# Layout
# --------------------------------------------------
col1, col2 = st.columns([1, 1], gap="large")

# --------------------------------------------------
# Upload Section
# --------------------------------------------------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📤 Upload X-ray Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Supported formats: JPG, PNG",
        type=["jpg", "jpeg", "png"]
    )

    original_image = None

    if uploaded_file:
        original_image = Image.open(uploaded_file).convert("RGB")
        original_image = resize_image(original_image)

        st.image(
            original_image,
            caption="Original X-ray Image",
            width=420
        )

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Detection Result Section
# --------------------------------------------------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Detection Results</div>', unsafe_allow_html=True)

    if uploaded_file and original_image:
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "input.jpg"
            Image.open(uploaded_file).convert("RGB").save(img_path)

            with st.spinner("Analyzing X-ray using deep learning model..."):
                results = model.predict(
                    source=str(img_path),
                    save=True,
                    conf=0.25
                )

            result_img_path = Path(results[0].save_dir) / "input.jpg"

            if result_img_path.exists():
                result_img = Image.open(result_img_path)
                result_img = resize_image(result_img)

                st.image(
                    result_img,
                    caption="Fracture Detection Output",
                    width=420
                )
            else:
                st.warning("Detection output could not be generated.")

    else:
        st.info("Upload an X-ray image to view detection results.")

    st.markdown('</div>', unsafe_allow_html=True)

# # --------------------------------------------------
# # Footer
# # --------------------------------------------------
# st.markdown("""
# <div class="footer">
#     Developed by <strong>Haseeb Iqbal</strong> · AI-based Medical Imaging 🧠<br>
#     For research & educational purposes only
# </div>
# """, unsafe_allow_html=True)
