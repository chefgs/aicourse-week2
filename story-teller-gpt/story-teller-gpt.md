# Story Teller GPT

## Overview
Story Teller GPT is an interactive Streamlit application for generating creative stories using various AI language models. This application provides a user-friendly interface to experiment with different text generation models, adjust parameters, and compare outputs side by side.

![Story Teller GPT Interface](https://example.com/storyteller_screenshot.png)

## Features

### Model Management
- **Model Selection**: Choose from a variety of open-source language models, including:
  - GPT-2 (various sizes: small, medium, large, xl)
  - DistilGPT2 (lightweight version)
  - Facebook's OPT-1.3B
- **Download Interface**: Download models directly from the UI before using them
- **Cache Detection**: Automatically detects which models are already downloaded and available

### Prompt Engineering
- **Ready-to-use Suggestions**: Choose from several story starters
- **Custom Prompts**: Write your own story beginning
- **Conclusion Control**: Option to instruct the AI to end stories with proper conclusions

### Generation Parameters
- **Max New Tokens**: Control the length of generated text
- **Number of Variations**: Generate multiple stories from the same prompt
- **Temperature**: Adjust the creativity vs. predictability of the output
- **Top-p (Nucleus Sampling)**: Control the diversity of token selection

## Technical Implementation

The application leverages Hugging Face's Transformers library to access state-of-the-art language models:

```python
# Core imports
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
```

### Efficient Model Loading
Models are loaded on demand, with a separate download step to prevent repeated downloads:

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

### Advanced Generation Technique
The app uses direct tokenizer and model access for precise control over generation:

```python
input_ids = tokenizer.encode(final_prompt, return_tensors='pt')
attention_mask = (input_ids != tokenizer.pad_token_id).long()
outputs = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=int(max_new_tokens),
    temperature=float(temperature),
    top_p=float(top_p),
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    num_return_sequences=int(num_return_sequences)
)
```

## Running the Application

### Prerequisites
- Python 3.7+
- Streamlit
- Transformers
- PyTorch

### Installation
```bash
pip install streamlit transformers torch
```

### Launch
```bash
streamlit run story_teller_gpt.py
```

## Educational Value

This application serves as both a creative writing tool and an educational resource for understanding:
- How language models generate text
- The effect of different parameters on text generation
- Differences between model sizes and architectures
- Prompt engineering techniques

## Tips for Best Results

- **Longer Prompts**: Generally lead to more focused and coherent stories
- **Temperature Settings**: 
  - Lower (0.1-0.5): More consistent, predictable outputs
  - Higher (0.8-1.2): More creative, surprising outputs
- **Model Selection**: Larger models (GPT-2 XL, OPT-1.3B) generally produce more coherent stories
- **Conclusion Instruction**: Works best with larger models that better follow instructions

## Future Enhancements

- Support for more models (BLOOM, LLaMA, etc.)
- Genre-specific prompt templates
- Character development options
- Story structure controls
- Export options for generated stories

---

Created with ❤️ using Streamlit and Hugging Face Transformers
