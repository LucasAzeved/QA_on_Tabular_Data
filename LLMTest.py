import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.2-3B"

# Baixa automaticamente o tokenizador e o modelo para uso
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Exemplo de prompt para teste de funcionamento
prompt = "Escreva uma expressão em pandas, que obtem o cabeçalho de um dataframe chamado df"

# Tokenize o prompt e prepare para o modelo
inputs = tokenizer(prompt, return_tensors="pt")

# Verificação do uso de CUDA
if torch.cuda.is_available():
    print("CUDA disponível:", torch.cuda.get_device_name(0))
    model = model.to("cuda")  # Move o modelo para a GPU
    inputs = {key: value.to("cuda") for key, value in inputs.items()}  # Move os inputs para a GPU

# Gera uma resposta a partir do modelo
outputs = model.generate(**inputs, max_length=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)