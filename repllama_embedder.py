import torch
import pickle
from transformers import AutoTokenizer, AutoModel
from peft import PeftConfig, PeftModel


torch.backends.cudnn.benchmark = True  # enable cudnn autotuner for optimized kernels

class ReplLlamaEmbedder:
    """
    A wrapper to embed documents using a RepLLaMA PEFT model,
    with mixed-precision and batching to mitigate OOM issues.
    """
    def __init__(
        self,
        peft_model_name: str,
        llm_model_name: str = 'meta-llama/Llama-2-7b-hf',
        max_length: int = 512,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16
    ):
        # device and dtype setup
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.max_length = max_length

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load and prepare model
        config = PeftConfig.from_pretrained(peft_model_name)
        base = AutoModel.from_pretrained(config.base_model_name_or_path).to(self.device)
        peft_model = PeftModel.from_pretrained(base, peft_model_name).merge_and_unload().to(self.device)
        peft_model.eval()
        # convert to half precision
        peft_model = peft_model.to(self.dtype)
        self.model = peft_model

    def embed_corpus(
        self,
        documents: list[str],
        batch_size: int = 4
    ) -> torch.Tensor:
        """
        Embed documents in batches with mixed precision to reduce memory footprint.

        Returns a tensor of shape (N, D).
        """
        all_emb = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i: i + batch_size]
            inputs = self.tokenizer(
                [f'passage: {doc}</s>' for doc in batch],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.dtype):
                outputs = self.model(**inputs)
                emb = outputs.last_hidden_state[:, -1, :]
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)

            all_emb.append(emb)
            # free any unused memory
            torch.cuda.empty_cache()

        return torch.cat(all_emb, dim=0)

    def get_feature_matrix(
        self,
        documents: list[str],
        batch_size: int = 4
    ) -> torch.Tensor:
        corpus = self.embed_corpus(documents, batch_size=batch_size)
        return corpus.T

    def random_projection(
        self,
        feature_matrix: torch.Tensor,
        n_features: int,
        seed: int = None
    ) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
        D_in, N = feature_matrix.shape
        R = torch.randn(n_features, D_in, device=feature_matrix.device, dtype=feature_matrix.dtype)
        return R @ feature_matrix

    def save_embeddings(self, embeddings: torch.Tensor, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(embeddings.cpu(), f)

