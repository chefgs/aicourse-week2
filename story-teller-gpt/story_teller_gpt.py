import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

st.set_page_config(page_title="story teller gpt", layout="wide")
st.title("Story Teller GPT")
st.markdown("""
A GenAI playground to experiment with story generation using various open-source models. Select models, adjust generation parameters, and compare outputs side by side.\

**Tip:** This app is designed to help you learn how GenAI text generation works!
""")

# Silence tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Model options
MODEL_OPTIONS = [
    "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "distilgpt2",
    "facebook/opt-1.3b"
]

with st.sidebar:
    st.header("User Tips")
    st.info("""
- **Model Selection:** Choose one or more models to compare their story generation styles.
- **Prompt:** Enter the beginning of your story. The models will continue from here.
- **Max Length:** Controls the total length of the output (including your prompt).
- **Number of Variations:** How many different stories to generate per model.
- **Top-p:** Controls output diversity. Lower values = more focused, higher = more creative.
""")

# Helper to check if model is downloaded (checks local cache)
def is_model_downloaded(model_name):
    from transformers.utils import cached_file, EntryNotFoundError
    try:
        # Try to find config file in cache
        cached_file(model_name, "config.json")
        return True
    except EntryNotFoundError:
        return False

# Download model UI
st.subheader("Model Download & Selection")
download_model = st.selectbox(
    "Select a model to download:", MODEL_OPTIONS, index=0,
    help="Download a model before using it for generation."
)
download_btn = st.button(f"Download '{download_model}' model")
if download_btn:
    with st.spinner(f"Downloading {download_model} ..."):
        try:
            AutoTokenizer.from_pretrained(download_model)
            AutoModelForCausalLM.from_pretrained(download_model)
            st.success(f"Model '{download_model}' downloaded and cached.")
        except Exception as e:
            st.error(f"Failed to download model '{download_model}': {e}")

# List of downloaded models
cached_models = [m for m in MODEL_OPTIONS if is_model_downloaded(m)]

# Model selection
selected_models = st.multiselect(
    "Select one or more downloaded models:",
    options=cached_models,
    default=[m for m in ["gpt2"] if "gpt2" in cached_models],
    help="Choose which models to use for story generation. Only downloaded models are shown."
)

# Task selection (for extensibility)
task = st.selectbox(
    "Select task:",
    options=["text-generation"],
    help="Choose the type of generation task. Only text-generation is supported in this demo."
)

# Prompt suggestions
PROMPT_SUGGESTIONS = [
    "Once upon a time in a quiet village, a young inventor discovered",
    "In the distant future, humanity made its greatest discovery when",
    "Deep in the enchanted forest, a mysterious light began to glow as",
    "A group of friends worked together and achieved their dreams, inspiring everyone around them.",
    "A young artist's kindness transformed their town into a place of joy and hope."
]

st.markdown("**Prompt Suggestions:**")
suggested_prompt = st.selectbox(
    "Choose a prompt suggestion (or ignore to write your own):",
    PROMPT_SUGGESTIONS,
    index=0,
    help="Select a prompt to auto-fill the prompt box below, or ignore to write your own."
)

# Prompt input
def_prompt = suggested_prompt if suggested_prompt else PROMPT_SUGGESTIONS[0]
prompt = st.text_area(
    "Enter your story prompt:",
    value=def_prompt,
    help="Type the beginning of your story. The model will continue from here."
)

# Generation parameters
col1, col2, col3 = st.columns(3)
with col1:
    max_new_tokens = st.number_input(
        "Max New Tokens", min_value=20, max_value=512, value=150,
        help="Number of tokens to generate beyond your prompt."
    )
with col2:
    num_return_sequences = st.number_input(
        "Number of Variations", min_value=1, max_value=5, value=1,
        help="Number of different story variations to generate."
    )
with col3:
    top_p = st.slider(
        "Top-p (nucleus sampling)", min_value=0.5, max_value=1.0, value=0.95, step=0.01,
        help="Controls the diversity of the output. Typical values: 0.8–1.0."
    )

temperature = st.slider(
    "Temperature (creativity)", min_value=0.1, max_value=1.5, value=0.8, step=0.05,
    help="Higher values = more creative, lower = more deterministic."
)

generate_btn = st.button("Generate Story")

if generate_btn:
    if not selected_models:
        st.warning("Please select at least one downloaded model.")
    elif not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        st.subheader("Generated Stories:")
        final_prompt = prompt
        for model_name in selected_models:
            with st.spinner(f"Loading model: {model_name}"):
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForCausalLM.from_pretrained(model_name)
                    input_ids = tokenizer.encode(final_prompt, return_tensors='pt')
                    
                    # Fix: Create attention_mask only if pad_token_id exists and is not None
                    attention_mask = None
                    if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                        attention_mask = (input_ids != tokenizer.pad_token_id).long()
                    
                    # Add special tokens to better guide the conclusion
                    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None
                    
                    # Fix: Only include attention_mask if it's not None
                    generate_kwargs = {
                        'input_ids': input_ids,
                        'max_new_tokens': int(max_new_tokens),
                        'temperature': float(temperature),
                        'top_p': float(top_p),
                        'do_sample': True,
                        'pad_token_id': tokenizer.eos_token_id,
                        'num_return_sequences': int(num_return_sequences),
                        'repetition_penalty': 1.0
                    }
                    
                    if attention_mask is not None:
                        generate_kwargs['attention_mask'] = attention_mask
                    
                    outputs = model.generate(**generate_kwargs)
                    st.markdown(f"**Model:** `{model_name}`")
                    for idx in range(outputs.shape[0]):
                        story = tokenizer.decode(outputs[idx], skip_special_tokens=True)
                        # Ensure the story ends at the last full stop
                        story = story.strip()
                        if not story.endswith('.'):
                            last_period = story.rfind('.')
                            if last_period != -1:
                                story = story[:last_period+1].strip()
                        st.success(story)
                except Exception as e:
                    st.error(f"Error loading or running model `{model_name}`: {e}")
