import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

st.set_page_config(page_title="story teller gpt", layout="wide")
st.title("Story Teller GPT")
st.markdown("""
A GenAI playground to experiment with story generation using various open-source models. Select models, adjust generation parameters, and compare outputs side by side.\

**Tip:** This app is designed to help you learn how GenAI text generation works!
""")

# Model options
MODEL_OPTIONS = [
    "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "distilgpt2",
    "mosaicml/mpt-7b-storywriter", "facebook/opt-1.3b", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "meta-llama/Llama-2-7b-chat-hf", "microsoft/phi-2", "tiiuae/falcon-7b-instruct",
    "deepseek-llm-7b", "deepseek-coder", "mpt-7b", "llama2-7b", "tinyllama"
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

# Model selection
selected_models = st.multiselect(
    "Select one or more models:",
    options=MODEL_OPTIONS,
    default=["gpt2"],
    help="Choose which models to use for story generation."
)

# Task selection (for extensibility)
task = st.selectbox(
    "Select task:",
    options=["text-generation"],
    help="Choose the type of generation task. Only text-generation is supported in this demo."
)

# Prompt input
prompt = st.text_area(
    "Enter your story prompt:",
    value="Once upon a time in a quiet village, a young inventor discovered",
    help="Type the beginning of your story. The model will continue from here."
)

# Generation parameters
col1, col2, col3 = st.columns(3)
with col1:
    max_length = st.number_input(
        "Max Length", min_value=20, max_value=512, value=150,
        help="Total length of the generated story including your prompt."
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
        st.warning("Please select at least one model.")
    elif not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        st.subheader("Generated Stories:")
        for model_name in selected_models:
            with st.spinner(f"Loading model: {model_name}"):
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForCausalLM.from_pretrained(model_name)
                    generator = pipeline(
                        task,
                        model=model,
                        tokenizer=tokenizer
                    )
                    results = generator(
                        prompt,
                        max_length=int(max_length),
                        num_return_sequences=int(num_return_sequences),
                        temperature=float(temperature),
                        top_p=float(top_p),
                        do_sample=True
                    )
                    st.markdown(f"**Model:** `{model_name}`")
                    for idx, res in enumerate(results):
                        st.success(res['generated_text'])
                except Exception as e:
                    st.error(f"Error loading or running model `{model_name}`: {e}")
