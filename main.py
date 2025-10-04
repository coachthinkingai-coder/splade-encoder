from fastapi import FastAPI, Body
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained("opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1")
model = AutoModelForMaskedLM.from_pretrained("opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1")
model.eval()

@app.post("/splade-embed")
def splade_embed(text: str = Body(..., embed=True)):
    with torch.no_grad():
        encoded = tokenizer(text, return_tensors="pt")
        outputs = model(**encoded)
        logits = outputs.logits
        # SPLADE: softplus(max over sequence dim)
        sparse_vector = torch.log(1 + torch.exp(torch.max(logits, dim=1).values)).squeeze()
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"].squeeze())
        input_ids = encoded["input_ids"].squeeze().tolist()
        result = {str(idx): float(s) for idx, s in zip(input_ids, sparse_vector.tolist())}
        return {"sparse_embedding": result}