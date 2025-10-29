from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("./models/trained_model")
model.push_to_hub("harpertoken/harpertokenGPT2")
