from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, AutoConfig, AutoModel
config = AutoConfig.from_pretrained("config.json")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = AutoModelForCausalLM.from_config(config)
model_inputs = tokenizer('I enjoy walking with my cute dog', return_tensors='pt')

# generate 40 new tokens
greedy_output = model.generate(**model_inputs, max_new_tokens=40)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

