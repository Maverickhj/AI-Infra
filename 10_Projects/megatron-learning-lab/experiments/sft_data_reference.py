"""Independent scalar Python data reference; authored tokens, no tokenizer/model."""
from __future__ import annotations
import json
import math
from pathlib import Path
import sys

FIXTURE = Path(__file__).resolve().parents[1]/'content/fixtures/sft-data.json'


def encode(fixture, sample, mode='assistant', limit=128):
    if mode not in {'assistant','last_turn','full'} or type(limit) is not int or not 2 <= limit <= 4096:
        raise ValueError('invalid mode or truncation limit')
    vocab = {t:i for i,t in enumerate(fixture['vocabulary'])}
    turns = sample['messages']
    last = max((i for i,m in enumerate(turns) if m['role']=='assistant'), default=-1)
    tokens = [(vocab['<bos>'], mode=='full')]
    for n, message in enumerate(turns):
        if ''.join(message['pieces']) != message['content']:
            raise ValueError('pieces must preserve raw messages')
        tokens.append((vocab['<'+message['role']+'>'], mode=='full'))
        response = message['role']=='assistant' and (mode=='assistant' or (mode=='last_turn' and n==last))
        tokens.extend((vocab[p], bool(response or mode=='full')) for p in message['pieces'])
        tokens.append((vocab['<eos>' if message['role']=='assistant' else '<turn>'], bool(response or mode=='full')))
    return tokens[:limit]


def calculate(fixture, indices, mode='assistant', limit=128, padding=1, leak=False, pad_to=0):
    if type(padding) is not int or not 1 <= padding <= 64 or not indices or len(set(indices)) != len(indices):
        raise ValueError('invalid padding or repeated/empty samples')
    ids, targets, labels, masks, positions, segments, valid = [],[],[],[],[],[],[]
    lengths, cu, physical = [],[0],[0]
    for index in indices:
        sample = fixture['samples'][index]
        tokens = encode(fixture,sample,mode,limit)
        lengths.append(len(tokens))
        width = pad_to or math.ceil(len(tokens)/padding)*padding
        if width < len(tokens): raise ValueError("padding cannot truncate")
        for pos in range(width):
            present = pos < len(tokens)
            target, selected = tokens[pos+1] if pos+1 < len(tokens) else (-100,False)
            ids.append(tokens[pos][0] if present else 0)
            targets.append(target)
            labels.append(target if selected else -100)
            masks.append(int(selected)); positions.append(pos)
            segments.append(sample['id']); valid.append(present)
        cu.append(cu[-1]+len(tokens)); physical.append(len(ids))
    allowed = [[bool(valid[i] and valid[j] and j<=i and (leak or segments[i]==segments[j])) for j in range(len(ids))] for i in range(len(ids))]
    contexts, losses = [],[]
    for i, token in enumerate(ids):
        if not valid[i]:
            contexts.append(0.0); losses.append(0.0); continue
        weights = [math.exp(token * key / 100) if allowed[i][j] else 0.0 for j,key in enumerate(ids)]
        context = math.fsum(w*key/7 for w,key in zip(weights,ids))/math.fsum(weights)
        logits = [math.cos((v+1)*(context+1)+positions[i]/3)/4 for v in range(len(fixture['vocabulary']))]
        log_partition = math.log(math.fsum(math.exp(z) for z in logits))
        contexts.append(context)
        losses.append(log_partition-logits[targets[i]] if targets[i]>=0 else 0.0)
    total = math.fsum(loss*mask for loss,mask in zip(losses,masks))
    count = sum(masks)
    return dict(ids=ids, targets=targets, labels=labels, masks=masks, positions=positions,
                segments=segments, valid=valid, lengths=lengths, cuSeqlens=cu, cuSeqlensPadded=physical, attention=allowed,
                contexts=contexts, losses=losses, sum=total, count=count,
                mean=total/count if count else None, status='reference' if count else 'no_supervision')


if __name__ == '__main__':
    fixture = json.loads(FIXTURE.read_text())
    requests = json.load(sys.stdin)
    print(json.dumps([calculate(fixture,**r) for r in requests], allow_nan=False))
