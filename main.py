from fastapi import FastAPI, Body
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained("opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1")
model = AutoModelForMaskedLM.from_pretrained("opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1")
model.eval()

@app.post("/splade-embed")
def splade_embed(text: str = Body(..., embed=True)):
    try:
        max_length = tokenizer.model_max_length  # Di solito 512
        input_ids = tokenizer.encode(text)
        if len(input_ids) > max_length:
            return {
                "error": f"Testo troppo lungo: {len(input_ids)} token. Massimo consentito: {max_length} token.",
                "status": "error"
            }

        # Encoding con troncatura sicura
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )
        outputs = model(**encoded)
        logits = outputs.logits
        # SPLADE: softplus(max over sequence dim)
        sparse_vector = torch.log(1 + torch.exp(torch.max(logits, dim=1).values)).squeeze()
        input_ids = encoded["input_ids"].squeeze().tolist()
        result = {str(idx): float(s) for idx, s in zip(input_ids, sparse_vector.tolist())}
        return {"sparse_embedding": result}
    except Exception as e:
        import logging
        logging.exception("Errore in /splade-embed")
        return {"error": str(e), "status": "error"}