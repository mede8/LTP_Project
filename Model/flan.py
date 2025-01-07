from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large", legacy=False)
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

input_text = "Write an emotionally rich story in a few paragraphs based on the Mona Lisa painting."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(
    input_ids,
    max_new_tokens=500,        
    num_beams=5,              
    temperature=0.8,          
    top_p=0.9,                
    repetition_penalty=2.0,    
    length_penalty=1.2,        
    no_repeat_ngram_size=3     
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
