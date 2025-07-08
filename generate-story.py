from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Choose a small model for demonstration; replace 'gpt2' with any causal LM like 'facebook/opt-1.3b'
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set up the generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Input prompt (start of your story)
prompt = "Once upon a time in a quiet village, a young inventor discovered"

# Generate story continuation
result = generator(
    prompt,
    max_length=150,         # Total length including the prompt
    num_return_sequences=1, # Number of stories to generate
    temperature=0.8,        # Creativity vs determinism
    top_p=0.95,             # Nucleus sampling
    do_sample=True          # Enables sampling (creative)
)

print("Generated Story:")
print(result[0]['generated_text'])
