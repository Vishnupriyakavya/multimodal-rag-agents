import numpy as np
from src.embed import embed_passages

sentences = [
    'Farmers get a subsidy for drip irrigation',
    'The government gives financial help to farmers for water-saving irrigation',
    'The cricket match starts at 7 pm',
]
a, b, c = [np.array(v) for v in embed_passages(sentences)]

print('Similar meaning  (1 vs 2):', round(float(a @ b), 3))
print('Unrelated        (1 vs 3):', round(float(a @ c), 3))