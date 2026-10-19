from src.ingest import col
from src.embed import embed_query
from src.llm import ask_llm

def retrieve(question, k=5):
    res = col.query(query_embeddings=[embed_query(question)], n_results=k)
    return [{'text': d, 'meta': m}
            for d, m in zip(res['documents'][0], res['metadatas'][0])]

def format_hit(h):
    m = h['meta']
    return '[' + m['source'] + ' p.' + str(m['page']) + ']\n' + h['text']

def answer(question, k=5):
    hits = retrieve(question, k)
    context = '\n\n'.join(format_hit(h) for h in hits)
    prompt = (
        'You are an assistant that answers ONLY from the context below.\n'
        'If the answer is not in the context, say: I could not find this in the documents.\n'
        'Mention the source file and page number for every fact.\n\n'
        'Context:\n' + context + '\n\n'
        'Question: ' + question + '\nAnswer:'
    )
    return ask_llm(prompt), hits