import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time

# Project Specific Imports
from CNN_encoder import CNN_Encoder
from configs import argHandler
from tokenizer_wrapper import TokenizerWrapper
from gpt2.gpt2_model import TFGPT2LMHeadModel
from predict import preprocess_image, generate_report, deduplicate_sentences
from grad_cam import GradCAMExplainer
from llm_translator import stream_translated_report

# --- Page Config ---
st.set_page_config(
    page_title="Chest X-Ray Report Generator",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling ---
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .report-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        font-family: 'Verdana', sans-serif;
        color: #000000;
    }
    .report-box ul {
        margin-top: 10px;
        padding-left: 20px;
    }
    .report-box li {
        margin-bottom: 8px;
        color: #003366; /* Dark Blue Text */
        font-size: 1.1em;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        height: 3em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- Constants ---
CHECKPOINT_DIRECTORY = r"checkpoints\CDGPT2\best_ckpt"
SUPPORTED_LANGUAGES = [
    "English", "Chinese", "Latin", "Arabic", "Hindi", "Spanish", "French", 
    "German", "Japanese", "Korean", "Russian", "Portuguese", "Italian", 
    "Turkish", "Indonesian", "Vietnamese"
]

# --- Resource Caching ---
# We cache these heavy resources so they are not reloaded on every interaction
@st.cache(allow_output_mutation=True)
def load_resources():
    # 1. Flags
    FLAGS = argHandler()
    FLAGS.setDefaults()
    FLAGS.ckpt_path = CHECKPOINT_DIRECTORY
    
    # 2. Tokenizer
    try:
        tokenizer_wrapper = TokenizerWrapper(
            FLAGS.all_data_csv,
            FLAGS.csv_label_columns[0],
            FLAGS.max_sequence_length,
            FLAGS.tokenizer_vocab_size
        )
    except FileNotFoundError:
        st.error("Tokenizer CSV not found. Please ensure `get_iu_xray.py` has been run.")
        return None, None, None, None

    # 3. Models
    encoder = CNN_Encoder(
        "pretrained_visual_model",
        FLAGS.visual_model_name,
        FLAGS.visual_model_pop_layers,
        FLAGS.encoder_layers,
        FLAGS.tags_threshold,
        num_tags=len(FLAGS.tags)
    )

    decoder = TFGPT2LMHeadModel.from_pretrained(
        "distilgpt2",
        from_pt=True,
        resume_download=True
    )

    # 4. Checkpoint
    optimizer = tf.keras.optimizers.Adam()
    ckpt = tf.train.Checkpoint(
        encoder=encoder,
        decoder=decoder,
        optimizer=optimizer
    )
    
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, FLAGS.ckpt_path, max_to_keep=1
    )

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print(f"Restored from {ckpt_manager.latest_checkpoint}")
    else:
        st.error(f"No checkpoint found in {FLAGS.ckpt_path}")
        return None, None, None, None

    return FLAGS, encoder, decoder, tokenizer_wrapper

# --- Main App ---
def main():
    st.title("🩻 Chest X-Ray Report Generator")
    st.markdown("Upload a Chest X-Ray image to generate a diagnostic report using **CDGPT2**.")

    # Sidebar
    st.sidebar.title("Configuration")
    st.sidebar.info("Model: CDGPT2 (DenseNet-121 + DistilGPT2)")
    st.sidebar.markdown("---")
    st.sidebar.text("Ensure checkpoints are\npresent in the\n`checkpoints` folder.")
    st.sidebar.markdown("---")
    st.sidebar.caption("Report will be cleaned and translated using DeepSeek V3")

    # Load Resources
    with st.spinner("Loading AI Models... This may take a moment."):
        FLAGS, encoder, decoder, tokenizer_wrapper = load_resources()

    if encoder is None:
        st.error("Failed to load models. Please check the logs.")
        return

    # Layout: Two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Upload Image")
        uploaded_file = st.file_uploader("Choose a Chest X-Ray", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            # Display Image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded X-Ray", width=350)
            
            # Language Settings
            st.subheader("2. Report Settings")
            selected_language = st.selectbox("Select Report Language", SUPPORTED_LANGUAGES, index=0)
            
            # Button to Generate
            if st.button("Generate Report 📝"):
                with col2:
                    st.subheader("3. Diagnostic Report")
                    with st.spinner("Analyzing image features and generating text..."):
                        
                        # Save temp file for consistency with preprocess_image logic if needed,
                        # but preprocess_image takes a path. Let's adapt or save temp.
                        # Since preprocess_image in predict.py takes path, let's save temp.
                        temp_path = "temp_uploaded_image.png"
                        image.save(temp_path)
                        
                        try:
                            # Preprocess
                            image_tensor = preprocess_image(temp_path)
                            
                            if image_tensor is not None:
                                start_time = time.time()
                                
                                # Generate
                                report = generate_report(FLAGS, encoder, decoder, tokenizer_wrapper, image_tensor)
                                
                                end_time = time.time()
                                
                                # --- DeepSeek V3 Translation & Cleaning ---
                                st.success(f"Initial Report Generated in {end_time - start_time:.2f}s. Sending to DeepSeek for translation...")
                                
                                report_placeholder = st.empty()
                                final_content = ""
                                
                                try:
                                    for chunk in stream_translated_report(report, selected_language):
                                        final_content += chunk
                                        formatted_content = final_content.replace('\n', '<br>')
                                        report_placeholder.markdown(f'<div class="report-box"><b>FINDINGS ({selected_language}):</b><br>{formatted_content}▌</div>', unsafe_allow_html=True)
                                        
                                    # Finalize (remove blinking cursor)
                                    if final_content:
                                        formatted_content = final_content.replace('\n', '<br>')
                                        report_placeholder.markdown(f'<div class="report-box"><b>FINDINGS ({selected_language}):</b><br>{formatted_content}</div>', unsafe_allow_html=True)
                                except Exception as e:
                                    st.error(f"DeepSeek API Error: {e}")

                                # --- Grad-CAM Visualization ---
                                with st.expander("Show AI Reasoning (Grad-CAM)"):
                                    with st.spinner("Generating Heatmap..."):
                                        try:
                                            # 1. Create Proxy Model for Grad-CAM (Target: Classification Output)
                                            # encoder.visual_model outputs [predictions, features]
                                            # We need a model that outputs just [predictions]
                                            proxy_model = tf.keras.models.Model(
                                                inputs=encoder.visual_model.input,
                                                outputs=encoder.visual_model.output[0]
                                            )
                                            
                                            # 2. Initialize Explainer
                                            explainer = GradCAMExplainer(proxy_model)
                                            
                                            # 3. Explain
                                            # We use the 'Normal' vs 'Abnormal' concept. 
                                            # If dataset has specific tags, we might want to target specific ones.
                                            # For now, let's target the top predicted tag.
                                            
                                            # Get prediction to find top class
                                            preds = proxy_model.predict(image_tensor)
                                            top_class_index = np.argmax(preds[0])
                                            
                                            # Generate Heatmap
                                            # We need to pass the numpy array of the image, not tensor, for consistency with module
                                            img_np = image_tensor.numpy()
                                            
                                            explanation = explainer.explain(
                                                image_input=img_np,
                                                class_index=top_class_index,
                                                method="gradcam++" # Use the advanced method
                                            )
                                            
                                            # Display
                                            st.image(explanation["superimposed_image"], caption=f"Attention Map (Top Class Index: {top_class_index})", width=350)
                                            
                                        except Exception as e:
                                            st.error(f"Could not generate Grad-CAM: {e}")

                                
                            else:
                                st.error("Error processing image.")
                        
                        except Exception as e:
                            st.error(f"An error occurred during generation: {e}")
                        finally:
                            # Cleanup
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
        else:
            with col2:
                st.info("👋 Upload an image to see the diagnosis here.")

if __name__ == "__main__":
    main()
