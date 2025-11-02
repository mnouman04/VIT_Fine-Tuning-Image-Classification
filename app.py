"""
🍔 ViT Food Classifier - Modern Streamlit App
A state-of-the-art food classification app using Vision Transformer
"""

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import io
import time
from pathlib import Path
import json

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="🍔 AI Food Classifier",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* Main background with gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Custom container styling */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Header styling */
    .app-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 800;
    }
    
    /* Subheader styling */
    .app-subheader {
        text-align: center;
        color: #2c3e50;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Card styling */
    .custom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }
    
    /* Prediction result styling */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
    }
    
    .prediction-label {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white;
    }
    
    .prediction-confidence {
        font-size: 1.5rem;
        opacity: 0.95;
        color: white;
    }
    
    /* Stats card */
    .stat-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: white;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.95;
        margin-top: 0.5rem;
        color: white;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
    }
    
    /* Upload box styling */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 1rem;
        background: white;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Info boxes */
    .info-box {
        background: #e8f4f8;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #2c3e50;
    }
    
    /* Markdown text visibility */
    .stMarkdown {
        color: #2c3e50;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #2c3e50;
        font-weight: 600;
    }
    
    [data-testid="stMetricLabel"] {
        color: #5a6c7d;
    }
    
    /* Success/Error/Info boxes */
    .stSuccess, .stError, .stInfo, .stWarning {
        background: white;
        color: #2c3e50;
        border-radius: 10px;
    }
    
    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        background: white;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Download button */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white !important;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(46, 204, 113, 0.4);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* H1, H2, H3 visibility */
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50 !important;
    }
    
    /* Paragraph text */
    p {
        color: #34495e;
    }
    
    /* Labels */
    label {
        color: #2c3e50 !important;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING AND CACHING
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_model_and_feature_extractor():
    """
    🔥 CRITICAL: This is where model and tokenizer loading happens!
    
    Models are expected to be in the root directory:
    - ./vit_food_classifier_hf/  (HuggingFace format with config.json, pytorch_model.bin, preprocessor_config.json)
    OR
    - ./vit_food_classifier_final.pt (single checkpoint file)
    
    For HuggingFace format (recommended):
    The directory should contain:
    - config.json
    - pytorch_model.bin or model.safetensors
    - preprocessor_config.json
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Try loading HuggingFace format first (RECOMMENDED)
        model_path = Path("./vit_food_classifier_hf")
        
        if model_path.exists():
            # 🔥 LOADING FEATURE EXTRACTOR (TOKENIZER) HERE
            feature_extractor = ViTFeatureExtractor.from_pretrained(str(model_path))
            
            # 🔥 LOADING MODEL HERE
            model = ViTForImageClassification.from_pretrained(str(model_path))
            model = model.to(device)
            model.eval()
            
            # Load class names if available
            class_names_file = model_path / "class_names.json"
            if class_names_file.exists():
                with open(class_names_file, 'r') as f:
                    class_names = json.load(f)
            else:
                # Default 101 food classes from Food-101 dataset
                class_names = load_default_class_names()
            
            return model, feature_extractor, class_names, device
        
        # Fallback: Try loading single checkpoint file
        else:
            checkpoint_path = Path("./vit_food_classifier_final.pt")
            if checkpoint_path.exists():
                # 🔥 LOADING FEATURE EXTRACTOR FROM PRETRAINED
                feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
                
                # Load class names
                class_names = load_default_class_names()
                num_classes = len(class_names)
                
                # 🔥 LOADING MODEL WITH CHECKPOINT
                model = ViTForImageClassification.from_pretrained(
                    'google/vit-base-patch16-224',
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True
                )
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                model = model.to(device)
                model.eval()
                
                return model, feature_extractor, class_names, device
            else:
                raise FileNotFoundError("Model files not found!")
                
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("""
        **Expected model files:**
        
        **Option 1 (Recommended):** HuggingFace format
        ```
        ./vit_food_classifier_hf/
        ├── config.json
        ├── pytorch_model.bin (or model.safetensors)
        ├── preprocessor_config.json
        └── class_names.json (optional)
        ```
        
        **Option 2:** Single checkpoint
        ```
        ./vit_food_classifier_final.pt
        ```
        """)
        raise

def load_default_class_names():
    """Load default Food-101 class names"""
    return [
        'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
        'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
        'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
        'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla',
        'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
        'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
        'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
        'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
        'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
        'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
        'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
        'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
        'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
        'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
        'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
        'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
        'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto',
        'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits',
        'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
        'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare',
        'waffles'
    ]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def preprocess_image(image, feature_extractor):
    """Preprocess image for model input"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Use feature extractor
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs['pixel_values']

def predict_image(model, image_tensor, device, class_names):
    """Make prediction on image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(pixel_values=image_tensor)
        logits = outputs.logits
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Get top 5 predictions
        top5_probs, top5_indices = torch.topk(probs, k=min(5, len(class_names)), dim=1)
        
        results = []
        for prob, idx in zip(top5_probs[0], top5_indices[0]):
            results.append({
                'class': class_names[idx.item()].replace('_', ' ').title(),
                'class_raw': class_names[idx.item()],
                'confidence': prob.item() * 100,
                'index': idx.item()
            })
        
        return results

def create_confidence_chart(predictions):
    """Create interactive confidence bar chart"""
    df = pd.DataFrame(predictions)
    
    fig = go.Figure()
    
    # Add gradient colors
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#ffeaa7']
    
    fig.add_trace(go.Bar(
        x=df['confidence'],
        y=df['class'],
        orientation='h',
        marker=dict(
            color=colors[:len(df)],
            line=dict(color='rgba(255,255,255,0.5)', width=2)
        ),
        text=df['confidence'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Confidence: %{x:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='🎯 Prediction Confidence',
            font=dict(size=20, color='#2c3e50', family='Arial Black')
        ),
        xaxis_title='Confidence (%)',
        yaxis_title='',
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        xaxis=dict(
            gridcolor='rgba(200,200,200,0.3)',
            range=[0, 100]
        ),
        yaxis=dict(
            categoryorder='total ascending'
        ),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_probability_gauge(confidence):
    """Create a gauge chart for confidence"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Level", 'font': {'size': 20, 'color': '#2c3e50'}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "darkblue"},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffcccc'},
                {'range': [50, 75], 'color': '#fff4cc'},
                {'range': [75, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#2c3e50", 'family': "Arial"}
    )
    
    return fig

def create_comparison_chart(predictions):
    """Create a radar chart comparing top predictions"""
    if len(predictions) < 3:
        return None
    
    categories = [p['class'] for p in predictions[:5]]
    values = [p['confidence'] for p in predictions[:5]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        line=dict(color='#667eea', width=2),
        marker=dict(size=8, color='#764ba2'),
        hovertemplate='<b>%{theta}</b><br>Confidence: %{r:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(200,200,200,0.3)'
            ),
            angularaxis=dict(
                gridcolor='rgba(200,200,200,0.3)'
            )
        ),
        showlegend=False,
        title=dict(
            text='📊 Multi-Class Comparison',
            font=dict(size=18, color='#2c3e50')
        ),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def get_food_emoji(food_name):
    """Get emoji for food category"""
    emoji_map = {
        'pizza': '🍕', 'burger': '🍔', 'hamburger': '🍔', 'sushi': '🍣',
        'taco': '🌮', 'burrito': '🌯', 'hot_dog': '🌭', 'sandwich': '🥪',
        'salad': '🥗', 'pasta': '🍝', 'spaghetti': '🍝', 'ramen': '🍜',
        'soup': '🍲', 'curry': '🍛', 'rice': '🍚', 'steak': '🥩',
        'chicken': '🍗', 'fish': '🐟', 'shrimp': '🍤', 'cake': '🍰',
        'ice_cream': '🍨', 'donut': '🍩', 'cookie': '🍪', 'chocolate': '🍫',
        'cheese': '🧀', 'bread': '🍞', 'croissant': '🥐', 'bagel': '🥯',
        'waffle': '🧇', 'pancake': '🥞', 'egg': '🍳', 'bacon': '🥓',
        'fries': '🍟', 'potato': '🥔', 'apple': '🍎', 'strawberry': '🍓'
    }
    
    food_lower = food_name.lower()
    for key, emoji in emoji_map.items():
        if key in food_lower:
            return emoji
    return '🍽️'

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="app-header">🍔 AI Food Classifier</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-subheader">Powered by Vision Transformer (ViT) | State-of-the-art Deep Learning</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        st.markdown("---")
        
        st.markdown("#### 📊 Model Information")
        st.info("""
        **Model**: Vision Transformer (ViT)
        
        **Architecture**: vit-base-patch16-224
        
        **Classes**: 101 Food Categories
        
        **Input Size**: 224x224 RGB
        """)
        
        st.markdown("---")
        st.markdown("#### 🎨 Features")
        st.success("""
        ✅ Real-time Classification
        
        ✅ Top-5 Predictions
        
        ✅ Confidence Visualization
        
        ✅ Interactive Charts
        
        ✅ Multi-image Support
        """)
        
        st.markdown("---")
        show_examples = st.checkbox("📸 Show Example Images", value=False)
        
        if show_examples:
            st.markdown("#### Sample Foods")
            st.markdown("""
            Try uploading images of:
            - 🍕 Pizza
            - 🍔 Hamburger
            - 🍣 Sushi
            - 🍰 Cake
            - 🥗 Salad
            """)
    
    # Load model
    with st.spinner('🔄 Loading AI Model...'):
        try:
            model, feature_extractor, class_names, device = load_model_and_feature_extractor()
            st.success('✅ Model Loaded Successfully!')
        except Exception as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            st.stop()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Food Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of food for classification"
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Image info
            st.markdown("#### 📋 Image Information")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Width", f"{image.size[0]}px")
            with col_b:
                st.metric("Height", f"{image.size[1]}px")
            with col_c:
                st.metric("Mode", image.mode)
    
    with col2:
        if uploaded_file is not None:
            st.markdown("### 🎯 Classification Results")
            
            # Classify button
            if st.button("🚀 Classify Food", use_container_width=True):
                with st.spinner('🔍 Analyzing image...'):
                    # Simulate processing time for effect
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.005)
                        progress_bar.progress(i + 1)
                    
                    # Preprocess and predict
                    image_tensor = preprocess_image(image, feature_extractor)
                    predictions = predict_image(model, image_tensor, device, class_names)
                    
                    # Store in session state
                    st.session_state['predictions'] = predictions
                    st.session_state['image'] = image
                
                st.success('✅ Classification Complete!')
    
    # Display results
    if 'predictions' in st.session_state:
        predictions = st.session_state['predictions']
        
        st.markdown("---")
        st.markdown("## 🏆 Prediction Results")
        
        # Top prediction highlight
        top_pred = predictions[0]
        emoji = get_food_emoji(top_pred['class_raw'])
        
        st.markdown(f"""
        <div class="prediction-box">
            <div style="font-size: 4rem; margin-bottom: 1rem;">{emoji}</div>
            <div class="prediction-label">{top_pred['class']}</div>
            <div class="prediction-confidence">{top_pred['confidence']:.2f}% Confidence</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed results
        st.markdown("---")
        
        # Three columns for visualizations
        viz_col1, viz_col2, viz_col3 = st.columns([2, 1, 2])
        
        with viz_col1:
            # Confidence bar chart
            fig_bar = create_confidence_chart(predictions)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with viz_col2:
            # Gauge chart
            fig_gauge = create_probability_gauge(top_pred['confidence'])
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with viz_col3:
            # Radar chart
            fig_radar = create_comparison_chart(predictions)
            if fig_radar:
                st.plotly_chart(fig_radar, use_container_width=True)
        
        # Detailed predictions table
        st.markdown("### 📊 Detailed Predictions")
        
        pred_df = pd.DataFrame([
            {
                'Rank': i+1,
                'Food Category': pred['class'],
                'Confidence (%)': f"{pred['confidence']:.2f}",
                'Emoji': get_food_emoji(pred['class_raw'])
            }
            for i, pred in enumerate(predictions)
        ])
        
        st.dataframe(
            pred_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Download results
        st.markdown("### 💾 Export Results")
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            # Export as JSON
            json_data = json.dumps(predictions, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name="predictions.json",
                mime="application/json"
            )
        
        with col_export2:
            # Export as CSV
            csv_data = pred_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="predictions.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p style='font-size: 1.1rem; margin-bottom: 0.5rem;'>
            <strong>🍔 AI Food Classifier</strong>
        </p>
        <p style='font-size: 0.9rem;'>
            Powered by Vision Transformer (ViT) & Streamlit
        </p>
        <p style='font-size: 0.8rem; margin-top: 1rem;'>
            Made with ❤️ using PyTorch & HuggingFace Transformers
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()