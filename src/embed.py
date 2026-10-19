from sentence_transformers import SentenceTransformer

_model = SentenceTransformer('intfloat/multilingual-e5-small')

def embed_passages(texts):
    # the e5 model expects these prefixes
    return _model.encode(['passage: ' + t for t in texts], normalize_embeddings=True).tolist()

def embed_query(query):
    return _model.encode(['query: ' + query], normalize_embeddings=True)[0].tolist()