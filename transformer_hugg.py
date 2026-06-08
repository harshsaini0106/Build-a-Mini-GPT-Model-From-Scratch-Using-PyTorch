from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = AutoModelForCausalLM.from_pretrained(
    "gpt2"
)

text = "Machine learning is"

inputs = tokenizer(
    text,
    return_tensors="pt"
)

outputs = model.generate(
    **inputs,
    max_length=50
)

print(
    tokenizer.decode(
        outputs[0]
    )
)