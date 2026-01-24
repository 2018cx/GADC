import torch
import argparse
from transformers import T5Tokenizer, T5EncoderModel
import gc
import json
from PIL import Image
import os

class TextEmbeddingProcessor:
    def __init__(self, model_name='google/flan-t5-xl',dtype="fp32"):
        self.model_name = model_name
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        self.model_dir = ''
        self.t5_model, self.t5_tokenizer = self.load_t5_model()


    def load_t5_model(self):
        tokenizer = T5Tokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        model = T5EncoderModel.from_pretrained(
            self.model_dir,
            local_files_only=True,
            torch_dtype=self.dtype,    
            low_cpu_mem_usage=True        
        )
        model.eval()
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        return model, tokenizer

    def compute_t5_embeddings(self, captions):
        inputs = self.t5_tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=16,
            return_tensors="pt",
            return_attention_mask=True
        ).to(self.t5_model.device)

        with torch.no_grad():
            outputs = self.t5_model(**inputs)
            embeddings = outputs.last_hidden_state.detach().cpu().to(torch.bfloat16)

        return embeddings, inputs["attention_mask"]

    def process_text_embeddings(self, captions):
        embeddings, attention_mask = self.compute_t5_embeddings(captions)

        return embeddings, attention_mask



def load_text_data(text_path):
    with open(text_path, 'r') as f:
        data = f.readlines()
    data = [i.strip() for i in data]
    data = [(i[:len("n09421951")], i[len("n09421951")+1:]) for i in data]
    return data




if __name__ == "__main__":

    processor = TextEmbeddingProcessor()
    json_path = "./caption.txt"
    extracted_data = load_text_data(json_path)
    root_path = "./caption_embeddings"
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    def prompt_template(class_name):
        return f"{class_name}"
    
    max_length = 0
    
    token_number_list = []
    for filename, class_name in extracted_data:
        prompt = prompt_template(class_name)
        embeddings, attention_mask = processor.process_text_embeddings(prompt)
        torch.save(embeddings, os.path.join(root_path, filename + ".pt"))
        torch.save(attention_mask, os.path.join(root_path, filename + "_mask.pt"))
        print(f"save {filename} to {root_path}")
        max_length = max(max_length, attention_mask.sum().item())
    print(max_length)


    print(f"Total samples: {len(token_number_list)}")
    print(f"Mean token count: {sum(token_number_list)/len(token_number_list):.2f}")
    print(f"Max token count: {max(token_number_list)}")
    print(f"Min token count: {min(token_number_list)}")