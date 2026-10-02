import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
os.makedirs(DATA, exist_ok=True)
rng = np.random.default_rng(0)
for i, name in enumerate(('AAA', 'BBB')):
    n = 8000
    toks = rng.integers(3, 256, size=n).astype(np.uint16)
    # sprinkle SEP (id=2) so the sentence-boundary paths are exercised
    toks[::37] = 2
    toks.tofile(os.path.join(DATA, f'token_stream_{name}_eos.bin'))
print('written', os.listdir(DATA))
