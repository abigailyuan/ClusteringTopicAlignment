import torch
import pickle
from transformers import AutoTokenizer, AutoModel

# Configuration
MODEL_NAME = "Agustinus/rap-llama"  # replace with the actual RapLlaMA model ID
OUTPUT_PATH = "wsj_rapllama_embeddings.pkl"
MAX_LENGTH = 512  # model's max token length

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


def embed_text(text: str) -> torch.Tensor:
    """
    Generate a mean-pooled embedding for a single text string.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    masked_states = hidden_states * mask
    summed = masked_states.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    embeddings = summed / counts

    return embeddings.squeeze(0).cpu()


def embed_text_list(text_list: list) -> list:
    """
    Embed a list of text strings. Returns a list of numpy embedding arrays.
    """
    embeddings = []
    for idx, text in enumerate(text_list):
        emb = embed_text(text)
        embeddings.append(emb.numpy())
        print(f"Embedded doc {idx} -> shape: {emb.shape}")
    return embeddings


if __name__ == "__main__":
    # Replace this with your WSJ dataset as a list of strings
    wsj_texts = [
        # "First article text...",
        # "Second article text...",
    ]

    embeddings = embed_text_list(wsj_texts)

    # Save embeddings to a pickle file
    with open(OUTPUT_PATH, "wb") as out_f:
        pickle.dump(embeddings, out_f)
    print(f"Saved embeddings to {OUTPUT_PATH}")
