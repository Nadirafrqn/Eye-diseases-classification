# main.py
import streamlit as st
from eye_disease_classification import klasifikasi_citra

def set_custom_style():
    """Apply custom styling to the app"""
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e9f2 100%);
        }
        .stButton>button {
            background-color: #2e70cc;
            color: white;
            border-radius: 10px;
            padding: 0.5rem 2rem;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #1e5ba8;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(46, 112, 204, 0.2);
        }
        .uploadedImage {
            border-radius: 15px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        .diagnosis-card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin: 1rem 0;
        }
        .sidebar .sidebar-content {
            background: #ffffff;
        }
        h1 {
            color: #1e3a8a;
            font-weight: 800;
            font-size: 2.5rem !important;
            margin-bottom: 2rem !important;
        }
        h2 {
            color: #2563eb;
            font-weight: 600;
        }
        .stAlert {
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    # Set page configuration
    st.set_page_config(
        page_title='Fundus Eye Disease Classifier',
        page_icon='👁️',
        layout='wide',
        initial_sidebar_state='expanded'
    )
    
    # Apply custom styling
    set_custom_style()
    
    # Add sidebar for navigation and information
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%); border-radius: 10px; margin-bottom: 1rem;'>
                <h1 style='color: white; font-size: 1.5rem; margin: 0;'>👁️ Eye Analysis</h1>
                <p style='color: #e2e8f0; font-size: 0.9rem; margin: 0.5rem 0 0 0;'>Fundus Image Classifier</p>
            </div>
        """, unsafe_allow_html=True)
        st.title("Navigation")
        st.markdown("---")
        
        st.markdown("""
        ### About Fundus Photography
        Fundus photography is a specialized form of medical imaging that captures detailed images 
        of the interior surface of the eye, including the retina, optic disc, macula, and posterior pole.
        
        ### Key Features
        - High-resolution imaging
        - Non-invasive procedure
        - Instant digital results
        - AI-powered analysis
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Supported Conditions")
        conditions = {
            "Cataract": "Clouding of the eye's natural lens",
            "Diabetic Retinopathy": "Diabetes-related blood vessel damage",
            "Glaucoma": "Optic nerve damage",
            "Normal": "Healthy eye condition"
        }
        
        for condition, description in conditions.items():
            st.markdown(f"**{condition}**")
            st.markdown(f"_{description}_")
            
    # Call the classification function
    klasifikasi_citra()

if __name__ == '__main__':
    main()

# klasifikasi_citra.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
# Remove plotly import as we'll use streamlit's native charts

@st.cache_resource
def load_model():
    """Load saved model and labels"""
    try:
        model = tf.keras.models.load_model('src/uap/final_model_cnn (1).h5')
        labels = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
        return model, labels
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        if len(img_array.shape) != 3 or img_array.shape[-1] != 3:
            raise ValueError(f"Invalid image shape: {img_array.shape}")
            
        return np.expand_dims(img_array, axis=0)
        
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return None

def create_results_card(label, probability, severity, color):
    """Create a styled card for displaying results"""
    background_color = {
        'High': 'rgba(239, 68, 68, 0.1)',
        'Moderate': 'rgba(245, 158, 11, 0.1)',
        'Low': 'rgba(34, 197, 94, 0.1)'
    }
    
    return f"""
        <div style="
            background: {background_color[severity]};
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            border-left: 4px solid {color};
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin: 0; color: {color}; font-size: 1.2rem;">{label}</h3>
                    <p style="margin: 0.2rem 0; color: #666;">Probability: {probability:.1f}%</p>
                </div>
                <div style="
                    background: {color};
                    color: white;
                    padding: 0.3rem 0.8rem;
                    border-radius: 15px;
                    font-size: 0.9rem;">
                    {severity}
                </div>
            </div>
        </div>
    """

def klasifikasi_citra():
    st.title('🔬 Fundus Eye Disease Analysis')
    
    model, labels = load_model()
    
    if model is None or labels is None:
        st.error("Model loading failed. Please check the model file.")
        return

    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    
    # Create two columns for layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
            ### Upload Fundus Image
            Please upload a clear, high-resolution fundus photograph for analysis.
        """)
        
        uploaded_file = st.file_uploader(
            "", 
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG",
            key='uploader'
        )
    
    if uploaded_file is None:
        st.session_state.predictions = None
        
        # Show sample images and guidelines
        st.markdown("### Guidelines for Best Results")
        cols = st.columns(3)
        with cols[0]:
            st.markdown("""
                #### 📸 Image Quality
                - High resolution
                - Good lighting
                - Clear focus
            """)
        with cols[1]:
            st.markdown("""
                #### 👁️ Positioning
                - Center the retina
                - Include optic disc
                - Proper orientation
            """)
        with cols[2]:
            st.markdown("""
                #### ⚠️ Avoid
                - Blurry images
                - Poor lighting
                - Partial views
            """)
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        with col1:
            st.image(image, caption='Uploaded Fundus Image', use_column_width=True)
        
        with col2:
            st.markdown("### Analysis Controls")
            analyze_button = st.button('🔍 Analyze Image', use_container_width=True)
            
            if analyze_button:
                st.session_state.predictions = None
                processed_image = preprocess_image(image)
                
                if processed_image is not None:
                    with st.spinner('Analyzing fundus image...'):
                        predictions = model.predict(processed_image, verbose=0)
                        st.session_state.predictions = predictions
        
        if st.session_state.predictions is not None:
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            predictions = st.session_state.predictions
            top_3_indices = predictions[0].argsort()[-3:][::-1]
            top_3_labels = [labels[idx].replace('_', ' ').title() for idx in top_3_indices]
            top_3_probs = predictions[0][top_3_indices]
            
            # Create columns for results
            res_col1, res_col2 = st.columns([3, 2])
            
            with res_col1:
                st.markdown("### Detailed Findings")
                for label, prob in zip(top_3_labels, top_3_probs):
                    severity = "High" if prob > 0.7 else "Moderate" if prob > 0.4 else "Low"
                    color = "#ef4444" if severity == "High" else "#f59e0b" if severity == "Moderate" else "#22c55e"
                    
                    st.markdown(
                        create_results_card(label, prob * 100, severity, color),
                        unsafe_allow_html=True
                    )
            
            with res_col2:
                st.markdown("### Probability Distribution")
                # Create DataFrame for visualization
                chart_data = pd.DataFrame({
                    'Condition': top_3_labels,
                    'Probability': top_3_probs * 100
                })
                # Use Streamlit's native bar chart
                st.bar_chart(
                    chart_data.set_index('Condition')['Probability'],
                    use_container_width=True
                )
            
            # Technical details in expander
            with st.expander("🔬 Technical Details"):
                st.markdown("""
                    ### Model Information
                    - Architecture: CNN
                    - Input Resolution: 224x224
                    - Color Space: RGB
                    """)
                st.markdown("### Raw Prediction Data")
                st.json({
                    'input_shape': processed_image.shape,
                    'value_range': {
                        'min': float(processed_image.min()),
                        'max': float(processed_image.max())
                    },
                    'raw_predictions': predictions[0].tolist(),
                    'probability_sum': float(np.sum(predictions[0]))
                })

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>🏥 This tool is for educational purposes only. Always consult healthcare professionals for medical diagnosis.</p>
        </div>
    """, unsafe_allow_html=True)