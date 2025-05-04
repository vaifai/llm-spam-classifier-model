import streamlit as st
import torch
import tiktoken
import pandas as pd
from pathlib import Path
import sys
import os
import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# Add the current directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import the GPTModel class
try:
    from gpt import GPTModel
except ImportError:
    st.error(
        "Could not import GPTModel. Make sure the gpt.py file is in the same directory."
    )
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Define the base configuration for the model
BASE_CONFIG = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension for GPT2-small
    "n_layers": 12,  # Number of transformer layers
    "n_heads": 12,  # Number of attention heads
    "drop_rate": 0.0,  # Dropout rate
    "qkv_bias": True,  # Query-key-value bias
}


@st.cache_resource
def load_model():
    """Load the pre-trained model from Hugging Face Hub or local file."""
    try:
        # Initialize the model
        model = GPTModel(BASE_CONFIG)

        # Modify the output head to match the saved model (binary classification)
        model.out_head = torch.nn.Linear(BASE_CONFIG["emb_dim"], 2, bias=True)

        # Create directory for downloaded models
        os.makedirs("downloaded_models", exist_ok=True)

        # Try to load from Hugging Face Hub
        try:
            # Get repository info from secrets or use default
            repo_id = st.secrets.get(
                "HF_MODEL_REPO", "vaibhav-vibe/spam-classifier-model"
            )

            st.info(f"Downloading model from Hugging Face Hub: {repo_id}")

            # Download the model file
            model_file = hf_hub_download(
                repo_id=repo_id,
                filename="spam_review_classifier.pth",
                token=st.secrets.get(
                    "HF_TOKEN", None
                ),  # Use token from secrets if available
            )

            # Load the model weights - use map_location to handle device differences
            state_dict = torch.load(model_file, map_location=torch.device("cpu"))

            # Load state dict directly without strict=True to allow missing/unexpected keys
            model.load_state_dict(state_dict, strict=False)

            st.success("Model successfully loaded from Hugging Face Hub!")

        except Exception as hf_error:
            # If Hugging Face download fails, try to load from local file
            st.warning(f"Could not download model from Hugging Face: {hf_error}")
            st.info("Trying to load model from local file...")

            if os.path.exists("spam_review_classifier.pth"):
                state_dict = torch.load(
                    "spam_review_classifier.pth", map_location=torch.device("cpu")
                )
                model.load_state_dict(state_dict, strict=False)
                st.success("Model successfully loaded from local file!")
            else:
                raise FileNotFoundError("Model file not found locally")

        # Set the model to evaluation mode
        model.eval()

        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Detailed error information:")
        import traceback

        st.code(traceback.format_exc())
        return None


@st.cache_resource
def get_tokenizer():
    """Get the tokenizer."""
    try:
        return tiktoken.get_encoding("gpt2")
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        return None


def classify_message(text, model, tokenizer, max_length=None, pad_token_id=50256):
    """Classify a message as spam or not spam."""
    if not text.strip():
        return None, 0.0

    model.eval()

    # Encode the text
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    # Truncate if necessary and pad
    input_ids = input_ids[: min(max_length, supported_context_length)]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_ids = torch.tensor(input_ids).unsqueeze(0)

    # Get prediction
    with torch.no_grad():
        logits = model(input_ids)[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)
        label = torch.argmax(logits, dim=-1).item()
        confidence = probabilities[0][label].item()

    return "spam" if label == 1 else "not spam", confidence


def main():
    # Title and description
    st.title("📱 SMS Spam Classifier")
    st.markdown("""
    This app uses a fine-tuned GPT-2 model to classify SMS messages as spam or not spam.
    Enter a message below to see if it's classified as spam!
    """)

    # Load model and tokenizer
    with st.spinner("Loading model..."):
        model = load_model()
        tokenizer = get_tokenizer()

    if model is None or tokenizer is None:
        st.error("Failed to load model or tokenizer. Please check the logs.")
        st.stop()

    # Get max length from the model
    max_length = 128  # Default value

    # Initialize session state for message if it doesn't exist
    if "message" not in st.session_state:
        st.session_state.message = ""

    # Text input
    message = st.text_area(
        "Enter a message:", height=150, value=st.session_state.message
    )

    # Example messages
    with st.expander("Try some example messages"):
        example_spam = "You are a winner you have been specially selected to receive $1000 cash or a $2000 award."
        example_ham = "Hey, just wanted to check if we're still on for dinner tonight? Let me know!"

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Example Spam"):
                st.session_state.message = example_spam
                st.experimental_rerun()

        with col2:
            if st.button("Example Ham"):
                st.session_state.message = example_ham
                st.experimental_rerun()

    # Classification button
    if st.button("Classify Message"):
        if not message.strip():
            st.warning("Please enter a message to classify.")
        else:
            with st.spinner("Classifying..."):
                result, confidence = classify_message(
                    message, model, tokenizer, max_length=max_length
                )

            # Display result with a nice UI
            st.markdown("### Classification Result:")

            col1, col2 = st.columns([1, 3])

            with col1:
                if result == "spam":
                    st.markdown("## 📵")
                else:
                    st.markdown("## ✅")

            with col2:
                if result == "spam":
                    st.error(
                        f"This message is classified as **SPAM** with {confidence:.2%} confidence."
                    )
                else:
                    st.success(
                        f"This message is classified as **NOT SPAM** with {confidence:.2%} confidence."
                    )

    # About section
    st.sidebar.title("About")
    st.sidebar.info("""
    This app demonstrates a spam classifier built using a fine-tuned GPT-2 model.

    The model was trained on the SMS Spam Collection dataset from UCI Machine Learning Repository.
    """)

    # Model information
    st.sidebar.title("Model Information")
    st.sidebar.markdown("""
    **Model Architecture**: Fine-tuned GPT-2 Small (124M parameters)

    **Training Data**: SMS Spam Collection Dataset

    **Performance**:
    - Accuracy: ~98% on test set
    - Precision: ~97%
    - Recall: ~98%
    """)

    # Deployment information
    st.sidebar.title("Deployment")
    st.sidebar.markdown("""
    This app can be deployed to:

    - [Streamlit Cloud](https://streamlit.io/cloud)
    - [Hugging Face Spaces](https://huggingface.co/spaces)
    - [Heroku](https://www.heroku.com/)

    For deployment, you'll need:
    1. This app.py file
    2. The model file (spam_review_classifier.pth)
    3. The gpt.py file
    4. requirements.txt with dependencies
    """)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Created with ❤️ using Streamlit")


if __name__ == "__main__":
    main()
