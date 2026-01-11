import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="EcoStyle - AI Fashion Assistant",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== STYLING ====================
st.markdown("""
    <style>
    .main-header {
        color: #2E86AB;
        text-align: center;
        padding: 20px 0;
    }
    .success-box {
        background-color: #D4EDDA;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28A745;
    }
    .info-box {
        background-color: #E7F3FF;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0066CC;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL & MAPPINGS ====================
@st.cache_resource
def load_model_and_mappings():
    """Load the trained model and label mappings"""
    try:
        base_dir = Path(__file__).resolve().parents[1]  # points to c:\Users\Sanjoli\ecostyle-project\src
        model_path = base_dir / "models" / "fashion_model_final.h5"
        mappings_path = base_dir / "models" / "label_mappings.json"
        
        # Check if files exist
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found at: {model_path}")
            st.info("Make sure fashion_model_final.h5 is in your models/ folder")
            return None, None
        
        if not os.path.exists(mappings_path):
            st.error(f"❌ Mappings file not found at: {mappings_path}")
            st.info("Make sure label_mappings.json is in your models/ folder")
            return None, None
        
        # Load model
        model = tf.keras.models.load_model(str(model_path))
        
        # Load mappings
        with open(mappings_path, 'r') as f:
            mappings = json.load(f)
        
        return model, mappings
    
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

# ==================== PREDICTION FUNCTION ====================
def predict_clothing(image, model, mappings):
    """
    Predict category and color of clothing item
    Returns: (category, color, cat_confidence, color_confidence)
    """
    try:
        # Preprocess image
        img = image.convert('RGB')  # Ensure RGB
        img = img.resize((224, 224))  # Model expects 224x224
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        
        # Extract results
        # Assuming model outputs [category_predictions, color_predictions]
        cat_probs = predictions[0][0]
        color_probs = predictions[1][0]
        
        cat_id = np.argmax(cat_probs)
        color_id = np.argmax(color_probs)
        
        cat_confidence = cat_probs[cat_id] * 100
        color_confidence = color_probs[color_id] * 100
        
        # Get names from mappings
        cat_name = mappings['id_to_category'][str(cat_id)]
        color_name = mappings['id_to_color'][str(color_id)]
        
        return cat_name, color_name, cat_confidence, color_confidence
    
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        return None, None, None, None

# ==================== MAIN APP ====================

# Sidebar
with st.sidebar:
    st.header("📱 EcoStyle")
    
    st.subheader("About")
    st.write("""
    **EcoStyle** is an AI-powered fashion assistant that helps you:
    - 👚 Identify clothing items
    - 📸 Organize your closet
    - 💰 Generate resale listings
    - ♻️ Promote sustainable fashion
    """)
    
    st.divider()
    
    st.subheader("📊 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Items in Closet", "0", "Coming Day 5")
    with col2:
        st.metric("Items Sold", "0", "Coming Day 6")
    
    st.divider()
    
    st.subheader("🗺️ Roadmap")
    st.write("""
    - ✅ **Day 3:** Model trained
    - ✅ **Day 4:** Basic app (Today!)
    - ⏳ **Day 5:** Virtual closet
    - ⏳ **Day 6:** Resale calculator
    - ⏳ **Day 7:** Deploy online
    """)
    
    st.divider()
    
    st.subheader("💡 Tips")
    st.write("""
    - Upload clear photos
    - Good lighting helps
    - Center the clothing item
    - Try different angles
    """)

# Main content
st.markdown("<h1 style='text-align: center; color: #2E86AB;'>👕 EcoStyle: AI Fashion Assistant</h1>", unsafe_allow_html=True)

st.write("""
<div style='text-align: center; font-size: 16px; color: #555; margin-bottom: 30px;'>
    Upload a photo of any clothing item and let AI identify it for you!
</div>
""", unsafe_allow_html=True)

# Load model
model, mappings = load_model_and_mappings()

if model is None or mappings is None:
    st.error("⚠️ Could not load model. Please check your file paths.")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Predict", "📚 Examples", "❓ Help"])

# ==================== TAB 1: PREDICT ====================
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose a clothing image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear photo of any clothing item"
        )
        
        if uploaded_file is not None:
            try:
                # Load and display image
                image = Image.open(uploaded_file)
                
                # Check image validity
                if image.mode not in ['RGB', 'RGBA', 'L']:
                    image = image.convert('RGB')
                
                st.image(image, caption='📷 Your Image', use_column_width=True)
                
                # Get image info
                st.caption(f"Size: {image.size} | Format: {image.format}")
                
            except Exception as e:
                st.error(f"❌ Error loading image: {e}")
                st.info("Please upload a valid image file (JPG or PNG)")
    
    with col2:
        st.subheader("🔍 Prediction Results")
        
        if uploaded_file is not None:
            try:
                # Make prediction
                with st.spinner('🤖 Analyzing your clothing item...'):
                    cat_name, color_name, cat_conf, color_conf = predict_clothing(
                        image, model, mappings
                    )
                
                if cat_name is not None:
                    # Display success
                    st.success("✅ Item Identified!", icon="✅")
                    st.divider()
                    
                    # Results cards
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        st.metric(
                            "📦 Category",
                            cat_name.upper(),
                            f"{cat_conf:.1f}% confidence"
                        )
                    
                    with result_col2:
                        st.metric(
                            "🎨 Color",
                            color_name.upper(),
                            f"{color_conf:.1f}% confidence"
                        )
                    
                    st.divider()
                    
                    # Action buttons
                    st.subheader("📋 What's Next?")
                    
                    action_col1, action_col2 = st.columns(2)
                    
                    with action_col1:
                        if st.button("➕ Add to Closet", key="add_closet"):
                            st.info("✨ Coming in Day 5: Virtual Closet feature!")
                    
                    with action_col2:
                        if st.button("💰 Sell This Item", key="sell_item"):
                            st.info("✨ Coming in Day 6: Resale Calculator feature!")
                    
                    # Confidence breakdown
                    with st.expander("📊 Confidence Breakdown"):
                        st.write(f"""
                        **Category Prediction:** {cat_conf:.2f}%
                        - Item identified as: **{cat_name}**
                        
                        **Color Prediction:** {color_conf:.2f}%
                        - Color identified as: **{color_name}**
                        
                        The model is confident about this prediction!
                        """)
                
            except Exception as e:
                st.error(f"❌ Prediction Error: {e}")
        
        else:
            st.info("👆 Upload an image to see predictions!")

# ==================== TAB 2: EXAMPLES ====================
with tab2:
    st.subheader("📚 How to Use EcoStyle")
    
    st.write("""
    ### ✅ Best Practices for Predictions
    
    **Good Photos:**
    - Clear, well-lit images
    - Clothing item centered in frame
    - Single item per photo
    - Neutral background (optional but helps)
    
    **Examples of Items You Can Upload:**
    - 👕 T-shirts
    - 👖 Jeans
    - 👗 Dresses
    - 👔 Jackets
    - 👞 Shoes
    - 👜 Bags
    - 🧣 Scarves
    - 🧤 Accessories
    
    ### 📸 Example Use Cases
    
    1. **Organizing Your Closet:**
       - Upload each item
       - Save to virtual closet
       - View collection by color/category
    
    2. **Selling on Resale Platforms:**
       - Upload photo
       - Get automatic listing
       - Post to Poshmark, Depop, etc.
    
    3. **Fashion Planning:**
       - See what colors you have
       - Identify gaps in wardrobe
       - Plan outfits
    """)
    
    # Sample code
    with st.expander("💻 Example Code Snippet"):
        st.code("""
# Using EcoStyle results
category = "T-Shirt"
color = "Blue"
confidence = "92%"

# You can:
# 1. Save to closet
# 2. Calculate resale price
# 3. Generate listing
# 4. Track wear frequency
        """, language="python")

# ==================== TAB 3: HELP ====================
with tab3:
    st.subheader("❓ Frequently Asked Questions")
    
    with st.expander("🤔 Why is my prediction wrong?"):
        st.write("""
        The AI model learns from examples. Common reasons for incorrect predictions:
        - **Poor lighting:** Make sure the image is well-lit
        - **Unclear item:** Wrinkled or unclear clothing is harder to identify
        - **Unusual style:** Very unique items may be misclassified
        - **Multiple items:** Upload one item at a time
        
        **Tip:** Try uploading again with better lighting!
        """)
    
    with st.expander("📱 Can I use this on my phone?"):
        st.write("""
        Yes! Streamlit Cloud (deploying in Day 7) will work on all devices:
        - 📱 iPhone/Android
        - 💻 Desktop
        - 🖥️ Laptop
        
        We'll deploy in Day 7!
        """)
    
    with st.expander("💾 Does it save my data?"):
        st.write("""
        **Day 4 (Current):** No, just predictions
        **Day 5:** Virtual closet will save your items locally
        **Day 7:** When deployed, items save to the cloud
        """)
    
    with st.expander("🎨 Why only certain colors?"):
        st.write("""
        The model was trained on specific color categories:
        - The more common colors (Red, Blue, Black, White, etc.)
        - Less common colors (Maroon, Teal) may be classified as closest match
        
        In Day 2, we trained the model on these specific categories!
        """)
    
    with st.expander("🚀 What's coming next?"):
        st.write("""
        **Day 5 (Virtual Closet):**
        - Save items from predictions
        - View your collection
        - Filter by color/category
        - Delete items
        
        **Day 6 (Resale Features):**
        - Calculate resale prices
        - Auto-generate listings
        - Copy to clipboard
        - Export for bulk upload
        
        **Day 7 (Deploy):**
        - Live on the internet
        - Share with friends
        - Full documentation
        - GitHub repository
        """)
    
    st.divider()
    
    st.subheader("📧 Feedback & Troubleshooting")
    
    with st.expander("🐛 I found a bug!"):
        st.write("""
        Please let me know! Common issues:
        - **Model not loading:** Check file paths in the code
        - **Image won't upload:** Try a different format (JPG instead of PNG)
        - **Slow predictions:** First prediction is slow (model loading), subsequent ones are fast
        """)
    
    with st.expander("📚 Want to learn more?"):
        st.write("""
        - **Streamlit Docs:** https://docs.streamlit.io
        - **TensorFlow Docs:** https://www.tensorflow.org
        - **Computer Vision:** https://www.cs231n.stanford.edu
        """)

# ==================== FOOTER ====================
st.divider()

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption("🚀 EcoStyle - Day 4 MVP")

with footer_col2:
    st.caption("AI Fashion Assistant")

with footer_col3:
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

st.caption("---")
st.caption("Built with ❤️ using Streamlit | Next: Day 5 - Virtual Closet Feature")