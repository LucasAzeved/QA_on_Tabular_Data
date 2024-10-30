from transformers import AutoModelForCausalLM, AutoTokenizer

# Nome do modelo no Hugging Face
model_name = "meta-llama/Llama-3.2-3B"

# Baixa automaticamente o tokenizador e o modelo para uso
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Exemplo de prompt para teste de funcionamento
prompt = "Explique em poucas palavras a importância da inteligência artificial."

# Tokenize o prompt e prepare para o modelo
inputs = tokenizer(prompt, return_tensors="pt")

# Gera uma resposta a partir do modelo
outputs = model.generate(**inputs, max_length=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)