import os

from molign.models import models_config


def test_embedding_methods():
    smiles = ["c1ccccc1", "CCCC"]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for emb_func, model_name, layer, kwargs in models_config:
        embs = emb_func(smiles, model_name, layer, **kwargs)
        print(model_name, layer, embs[0:5, 0:5], embs.shape)

if __name__ == "__main__":
    test_embedding_methods()