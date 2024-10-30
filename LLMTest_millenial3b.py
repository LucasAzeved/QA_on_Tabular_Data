
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("budecosystem/code-millenials-3b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("budecosystem/code-millenials-3b")

template = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
### Instruction: {instruction} ### Response:"""

instruction = "Write a Python function to calculate the factorial of a number."

prompt = template.format(instruction=instruction)

inputs = tokenizer(prompt, return_tensors="pt")
sample = model.generate(**inputs, max_length=128)
print(tokenizer.decode(sample[0]))
