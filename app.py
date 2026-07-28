import os
import gc
import streamlit as st
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# PyTorch Optimization
torch.set_num_threads(1)

# ---------------------------------------------------------
# 🎯 1. DATASET CLASSES & HIGH-PRECISION SUB-TYPES
# ---------------------------------------------------------
DATASET_CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer", 
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

SUB_CLASSES = {
    "Horse": ["Arabian Horse", "Thoroughbred Horse", "Quarter Horse", "Appaloosa Horse", "Friesian Horse", "Clydesdale Horse", "Mustang Horse", "Pony", "Sorrel Horse", "Chestnut Horse", "Stallion"],
    "Dog": ["Labrador Retriever", "German Shepherd", "Golden Retriever", "Bulldog", "Beagle", "Poodle", "Rottweiler", "Saluki Hound", "Siberian Husky", "Pug", "Doberman Pinscher", "Boxer Dog"],
    "Cat": ["Persian Cat", "Siamese Cat", "Maine Coon Cat", "Bengal Cat", "Sphynx Cat", "British Shorthair Cat", "Ragdoll Cat", "Tabby Cat", "Scottish Fold Cat"],
    "Bird": ["Parrot", "Eagle", "Falcon", "Owl", "Peacock", "Flamingo", "Sparrow", "Pigeon", "Canary", "Macaw", "Kingfisher", "Toucan"],
    "Automobile": ["Sedan Car", "SUV Automobile", "Sports Car", "Hatchback Car", "Convertible Car", "Coupe Car", "Limousine", "Vintage Car", "Race Car"],
    "Truck": ["Pickup Truck", "Semi Truck", "Monster Truck", "Dump Truck", "Fire Truck", "Delivery Van", "Tow Truck"],
    "Airplane": ["Commercial Passenger Airliner", "Fighter Jet", "Helicopter", "Propeller Airplane", "Cargo Airplane", "Biplane"],
    "Ship": ["Cruise Ship", "Cargo Ship", "Luxury Yacht", "Sailboat", "Submarine", "Speedboat", "Container Ship", "Warship"],
    "Deer": ["White-tailed Deer", "Reindeer", "Elk Deer", "Moose", "Fallow Deer", "Red Deer"],
    "Frog": ["Tree Frog", "Bullfrog", "Poison Dart Frog", "Toad", "Green Tree Frog"]
}

# ---------------------------------------------------------
# 2. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Prism Vision",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 3. SIDEBAR CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 15px 10px; background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 16px; backdrop-filter: blur(12px); box-shadow: 0 0 20px rgba(56, 189, 248, 0.2); margin-bottom: 20px;'>
            <h2 style='color: #ffffff; font-weight: 900; font-size: 1.5rem; margin: 0; letter-spacing: 1.5px; text-shadow: 0 0 12px rgba(56, 189, 248, 0.8);'>⚙️ CONTROL PANEL</h2>
            <p style='color: #94a3b8; font-size: 0.8rem; margin-top: 6px; font-weight: 500;'>Ultra Metallic Controls</p>
        </div>
    """, unsafe_allow_html=True)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.08) 100%); border: 1px solid rgba(52, 211, 153, 0.6); padding: 14px; border-radius: 14px; text-align: center; margin-bottom: 20px; box-shadow: 0 0 18px rgba(52, 211, 153, 0.3);'>
            <p style='margin:0; color: #34d399; font-weight: 800; font-size: 0.85rem; letter-spacing: 1px; text-shadow: 0 0 8px rgba(52, 211, 153, 0.8);'>🚀 GPU ACCELERATED</p>
            <p style='margin:4px 0 0 0; color: #a7f3d0; font-size: 0.8rem; font-weight: 500;'>{gpu_name}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.08) 100%); border: 1px solid rgba(248, 113, 113, 0.6); padding: 14px; border-radius: 14px; text-align: center; margin-bottom: 20px;'>
            <p style='margin:0; color: #f87171; font-weight: 800; font-size: 0.85rem; letter-spacing: 1px; text-shadow: 0 0 8px rgba(248, 113, 113, 0.8);'>⚠️ RUNNING ON CPU</p>
            <p style='margin:4px 0 0 0; color: #fca5a5; font-size: 0.8rem; font-weight: 500;'>PyTorch GPU support not detected</p>
        </div>
        """, unsafe_allow_html=True)

    theme_mode = st.selectbox(
        "🎨 Select Animated Background",
        ["Cyber Blue Moving Grid", "Neon Green Matrix Stream", "Purple Cosmic Motion"],
        index=0
    )
    
    st.markdown("<hr style='border: 0; height: 1px; background: linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.3) 50%, rgba(255,255,255,0) 100%); margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #38bdf8; font-size: 1.05rem; font-weight: 800; margin-bottom: 12px; text-shadow: 0 0 10px rgba(56, 189, 248, 0.6);'>🏷️ Target Classes (10)</h3>", unsafe_allow_html=True)
    
    class_tags = "".join([f"<span style='background: rgba(255,255,255,0.06); color: #e2e8f0; border: 1px solid rgba(255, 255, 255, 0.25); padding: 5px 11px; border-radius: 8px; margin: 3px; display: inline-block; font-size: 0.8rem; font-weight: 600; box-shadow: inset 0 1px 0 rgba(255,255,255,0.3);'>{cls}</span>" for cls in DATASET_CLASSES])
    st.markdown(f"""
    <div style='background: rgba(10, 15, 25, 0.7); padding: 14px; border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.15); backdrop-filter: blur(14px); box-shadow: inset 0 0 15px rgba(0, 0, 0, 0.5);'>
        {class_tags}
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. CONTINUOUS MOVING BACKGROUND & HIGH-GLOW BUTTON CSS
# ---------------------------------------------------------
if theme_mode == "Neon Green Matrix Stream":
    accent_hex = "#10b981"
    grid_color = "rgba(16, 185, 129, 0.18)"
    particle_color = "rgba(52, 211, 153, 0.35)"
    btn_grad = "linear-gradient(135deg, #059669 0%, #10b981 50%, #047857 100%)"
    glow_shadow = "rgba(16, 185, 129, 0.6)"
elif theme_mode == "Purple Cosmic Motion":
    accent_hex = "#c084fc"
    grid_color = "rgba(192, 132, 252, 0.18)"
    particle_color = "rgba(168, 85, 247, 0.35)"
    btn_grad = "linear-gradient(135deg, #7e22ce 0%, #a855f7 50%, #581c87 100%)"
    glow_shadow = "rgba(192, 132, 252, 0.6)"
else:  # Cyber Blue Moving Grid
    accent_hex = "#38bdf8"
    grid_color = "rgba(56, 189, 248, 0.18)"
    particle_color = "rgba(14, 165, 233, 0.35)"
    btn_grad = "linear-gradient(135deg, #0284c7 0%, #38bdf8 50%, #1e40af 100%)"
    glow_shadow = "rgba(56, 189, 248, 0.6)"

st.markdown(f"""
    <style>
        /* TRANSPARENT CONTAINERS */
        .stApp, [data-testid="stHeader"], [data-testid="stToolbar"], .main, .block-container {{
            background: transparent !important;
        }}

        /* CONTINUOUS MOVING BACKGROUND */
        [data-testid="stAppViewContainer"] {{
            background-color: #02040a !important;
            background-image: 
                radial-gradient(circle at 50% 50%, {particle_color} 1px, transparent 2px),
                linear-gradient(to right, {grid_color} 1px, transparent 1px),
                linear-gradient(to bottom, {grid_color} 1px, transparent 1px) !important;
            background-size: 40px 40px, 50px 50px, 50px 50px !important;
            animation: continuousBackgroundMove 10s linear infinite !important;
        }}

        @keyframes continuousBackgroundMove {{
            0% {{ background-position: 0px 0px, 0px 0px, 0px 0px; }}
            100% {{ background-position: 400px 800px, 500px 500px, 500px 500px; }}
        }}

        /* HERO TYPOGRAPHY */
        .hero-title {{
            font-size: 3.4rem; font-weight: 900;
            background: linear-gradient(90deg, #ffffff 0%, {accent_hex} 50%, #ffffff 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            text-align: center; margin-bottom: 0px;
            letter-spacing: 1.5px;
            filter: drop-shadow(0 0 15px {accent_hex});
        }}
        .hero-subtitle {{
            text-align: center; color: #94a3b8; font-size: 1.15rem; margin-bottom: 2.5rem; font-weight: 500;
        }}
        .glow-text-header {{
            color: #ffffff !important;
            font-weight: 800 !important;
            letter-spacing: 0.5px;
        }}

        /* GLASSMORPHISM CARDS */
        .glass-card {{
            background: rgba(8, 14, 28, 0.75) !important;
            border-radius: 22px; 
            padding: 26px; 
            backdrop-filter: blur(25px) saturate(180%);
            -webkit-backdrop-filter: blur(25px) saturate(180%);
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.22);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.95), 
                        inset 0 1px 1px rgba(255, 255, 255, 0.4);
            transition: all 0.4s ease;
            margin-bottom: 22px;
        }}

        /* ULTRA SHINING & PULSING BUTTON STYLING */
        div.stButton > button {{
            position: relative !important;
            overflow: hidden !important;
            background: {btn_grad} !important; 
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.6) !important; 
            border-radius: 14px !important;
            padding: 18px 30px !important; 
            font-weight: 900 !important;
            font-size: 1.1rem !important; 
            width: 100%; 
            letter-spacing: 1.5px !important;
            text-transform: uppercase;
            cursor: pointer;
            z-index: 1;
            box-shadow: 0 0 20px {glow_shadow}, inset 0 1px 2px rgba(255, 255, 255, 0.8) !important;
            animation: buttonPulse 2.5s infinite ease-in-out !important;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        }}

        @keyframes buttonPulse {{
            0% {{ box-shadow: 0 0 15px {glow_shadow}, inset 0 1px 2px rgba(255, 255, 255, 0.8); }}
            50% {{ box-shadow: 0 0 35px {glow_shadow}, 0 0 10px #ffffff, inset 0 1px 4px rgba(255, 255, 255, 0.9); }}
            100% {{ box-shadow: 0 0 15px {glow_shadow}, inset 0 1px 2px rgba(255, 255, 255, 0.8); }}
        }}

        div.stButton > button::before {{
            content: '' !important;
            position: absolute !important;
            top: -50% !important;
            left: -150% !important;
            width: 70% !important;
            height: 200% !important;
            background: linear-gradient(
                90deg, 
                rgba(255, 255, 255, 0) 0%, 
                rgba(255, 255, 255, 0.8) 50%, 
                rgba(255, 255, 255, 0) 100%
            ) !important;
            transform: rotate(25deg) !important;
            animation: continuousButtonShine 2.5s infinite linear !important;
            z-index: 2;
        }}

        @keyframes continuousButtonShine {{
            0% {{ left: -150%; }}
            50% {{ left: 150%; }}
            100% {{ left: 150%; }}
        }}

        div.stButton > button:hover {{
            transform: translateY(-3px) scale(1.02) !important;
            filter: brightness(1.3) !important;
            border-color: #ffffff !important;
            box-shadow: 0 15px 40px {glow_shadow}, 0 0 25px rgba(255, 255, 255, 0.9) !important;
        }}

        div.stButton > button:active {{
            transform: translateY(1px) scale(0.98) !important;
            filter: brightness(0.9) !important;
        }}

        /* TABS STYLING */
        .stTabs [data-baseweb="tab-list"] {{ 
            gap: 16px; 
            background: rgba(5, 8, 18, 0.85); 
            padding: 10px 14px;
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(20px);
        }}
        
        .stTabs [data-baseweb="tab"] {{
            height: 54px; border-radius: 12px; padding: 0 26px;
            color: #94a3b8 !important; font-weight: 700;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            transition: all 0.3s ease !important;
        }}

        .stTabs [aria-selected="true"] {{
            background: {btn_grad} !important; color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.5) !important;
        }}

        /* SIDEBAR STYLING */
        section[data-testid="stSidebar"] {{
            background: rgba(2, 4, 10, 0.92) !important;
            backdrop-filter: blur(25px) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        }}

        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. LOAD CLIP MODEL
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device

model, processor, device = load_clip_model()

# ---------------------------------------------------------
# 6. ENSEMBLED PREDICTION ENGINES
# ---------------------------------------------------------
def predict_hierarchical(image):
    """ Detailed multi-prompt ensembling for single inference """
    main_prompts = [f"a photo of a {cls.lower()}" for cls in DATASET_CLASSES]
    inputs_main = processor(text=main_prompts, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs_main = model(**inputs_main)
        probs_main = outputs_main.logits_per_image.softmax(dim=1)[0].cpu().numpy() * 100

    main_results = sorted(zip(DATASET_CLASSES, probs_main), key=lambda x: x[1], reverse=True)
    detected_main = main_results[0][0]
    top_main_class = detected_main.upper()
    top_main_conf = main_results[0][1]

    main_names = [r[0] for r in main_results]
    main_probs = [r[1] for r in main_results]

    sub_list = SUB_CLASSES.get(detected_main, [detected_main])
    templates = [
        "a photo of a {}",
        "a clear photo of a {}",
        "a close-up photo of a {}",
        "a picture showing a {}"
    ]

    ensemble_probs = np.zeros(len(sub_list))

    for tmpl in templates:
        sub_prompts = [tmpl.format(sub.lower()) for sub in sub_list]
        inputs_sub = processor(text=sub_prompts, images=image, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs_sub = model(**inputs_sub)
            probs_sub = outputs_sub.logits_per_image.softmax(dim=1)[0].cpu().numpy()
            ensemble_probs += probs_sub

    final_sub_probs = (ensemble_probs / len(templates)) * 100
    sub_results = sorted(zip(sub_list, final_sub_probs), key=lambda x: x[1], reverse=True)
    top_sub_type = sub_results[0][0]

    return top_main_class, top_main_conf, top_sub_type, main_names, main_probs


st.markdown("<h1 class='hero-title'>💎 PRISM VISION</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>Deep Visual Intelligence & High-Precision Category Detection</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🎯 Classifier Workspace", "📊 Performance Heatmap"])

# ----------------- TAB 1: SINGLE INFERENCE -----------------
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='glow-text-header' style='font-size: 1.3rem; margin-bottom: 16px;'>🖼️ Input Workspace</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Image (JPG, PNG, WEBP)...", type=["jpg", "png", "jpeg", "webp"], key="single")
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image Preview", width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='glow-text-header' style='font-size: 1.3rem; margin-bottom: 16px;'>🎯 Inference Results</h3>", unsafe_allow_html=True)
        
        if uploaded_file and st.button("🚀 EXECUTE PREDICTION", key="btn_single"):
            with st.spinner("⚡ Running Ensembled High-Precision Model..."):
                try:
                    main_cls, main_conf, sub_type, class_names, class_probs = predict_hierarchical(image)
                    
                    st.markdown(f"""
                    <div style='text-align:center; padding: 26px; background: rgba(255, 255, 255, 0.04); border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.25); box-shadow: 0 0 30px rgba(0, 0, 0, 0.8); backdrop-filter: blur(15px);'>
                        <p style='margin:0; font-weight:800; color: #94a3b8; font-size: 0.85rem; letter-spacing: 2px;'>MAIN DATASET CLASS</p>
                        <h1 style='font-size:3.2rem; margin:6px 0; color:#38bdf8; font-weight:900; letter-spacing: 1px; text-shadow: 0 0 18px {accent_hex};'>{main_cls}</h1>
                        <p style='margin:10px 0 4px 0; font-weight:600; color: #cbd5e1; font-size: 1.1rem;'>Detected Sub-type / Breed: <b style='color:#c084fc; font-weight: 800; text-shadow: 0 0 12px rgba(192, 132, 252, 0.7);'>{sub_type}</b></p>
                        <h3 style='color:#34d399; margin:6px 0 0 0; font-weight:800; font-size:1.25rem; text-shadow: 0 0 12px rgba(52, 211, 153, 0.7);'>Class Confidence: {main_conf:.2f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("")
                    st.markdown("<h4 class='glow-text-header' style='font-size: 1.1rem; margin-top: 15px;'>📊 Main Classes Probability Distribution</h4>", unsafe_allow_html=True)
                    
                    fig = go.Figure(go.Bar(
                        x=class_probs[::-1], y=class_names[::-1], orientation='h',
                        marker=dict(color=class_probs[::-1], colorscale='Turbo')
                    ))
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#f8fafc', family="Arial"),
                        margin=dict(l=10, r=10, t=10, b=10), height=340
                    )
                    st.plotly_chart(fig, width='stretch')
                except Exception as e:
                    st.error(f"Execution Error: {str(e)}")
        else:
            st.info("👈 Upload an image on the left workspace.")
        st.markdown("</div>", unsafe_allow_html=True)


# ----------------- TAB 2: MODEL MATRIX -----------------
with tab2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='glow-text-header' style='font-size: 1.3rem; margin-bottom: 16px;'>📊 Confusion Matrix Simulation</h3>", unsafe_allow_html=True)
    
    dummy_cm = np.random.randint(10, 990, size=(len(DATASET_CLASSES), len(DATASET_CLASSES)))
    fig_cm = px.imshow(
        dummy_cm, x=DATASET_CLASSES, y=DATASET_CLASSES,
        text_auto=True, color_continuous_scale='Viridis',
        labels=dict(x="Predicted Class", y="Actual Class", color="Count")
    )
    fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#f8fafc'), height=480)
    st.plotly_chart(fig_cm, width='stretch')
    st.markdown("</div>", unsafe_allow_html=True)
