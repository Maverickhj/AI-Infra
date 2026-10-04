"""Independent scalar CPU GQA reference; no frontend, numpy, torch or checkpoints.

B=1 is represented by omitting the batch axis. Matrices use logical [in,out]
weights; mixed QKV storage is [Q0,Q1,K0,V0,Q2,Q3,K1,V1] for this fixture.
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'content/fixtures/gqa-reference.json'


def authored_fixture():
    def matrix(rows, cols, salt):
        return [[(((i + 1) * 7 + (j + 1) * 11 + salt * (i + 2)) % 23 - 11) / 17
                 for j in range(cols)] for i in range(rows)]
    return dict(schema_version=1, evidence='reference',
                architecture_origin='authored_scaled_gqa', weights_origin='authored_fixture',
                training_execution='not_run', dtype='float64',
                dimensions=dict(B=1, S=4, H=8, nq=4, nkv=2, d=4),
                epsilon=1e-6, theta=10000, positions=[0, 1, 2, 3],
                rotary_layout='split_half', mask_convention='true_means_allowed; key_index <= query_index',
                weight_layout='logical_in_out; grouped_QKV: Q0,Q1,K0,V0,Q2,Q3,K1,V1',
                x=matrix(4, 8, 1), wqkv=matrix(8, 32, 2), wo=matrix(16, 8, 3),
                input_gain=[1 + i / 20 for i in range(8)],
                q_gain=[1, 1.1, .9, 1.2], k_gain=[.95, 1.05, 1.15, .85],
                qkv_bias=[(i % 7 - 3) / 20 for i in range(32)],
                output_bias=[0] * 8,
                variants={'qwen3': {'qk_norm': True, 'qkv_bias': False},
                          'qwen25': {'qk_norm': False, 'qkv_bias': True}})


def validate(f):
    dims = f.get('dimensions', {})
    size = dims.get('S')
    if any(dims.get(k) != v for k,v in dict(B=1,H=8,nq=4,nkv=2,d=4).items()) or type(size) is not int or not 1 <= size <= 64:
        raise ValueError('Supported: B1/H8/nq4/nkv2/d4 with integer sequence length 1–64')
    if f.get('rotary_layout') != 'split_half':
        raise ValueError('RoPE layout must be split_half')
    def check(value, shape):
        if shape:
            if not isinstance(value, list) or len(value) != shape[0]:
                raise ValueError('Invalid tensor shape or empty input')
            for child in value:
                check(child, shape[1:])
        elif type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Tensor entries must be finite numbers')
    for key, shape in [('x', [size,8]), ('wqkv', [8,32]), ('wo', [16,8]),
                       ('input_gain',[8]), ('q_gain',[4]), ('k_gain',[4]),
                       ('qkv_bias',[32]), ('output_bias',[8]), ('positions',[size])]:
        check(f.get(key), shape)
    for key in ('epsilon', 'theta'):
        if type(f.get(key)) not in (int, float) or not math.isfinite(f[key]) or f[key] <= 0:
            raise ValueError(f'{key} must be positive and finite')
    if any(type(p) is not int or p < 0 for p in f['positions']):
        raise ValueError('Position IDs must be nonnegative integers')


def validate_selection(head, token):
    if type(head) is not int or not 0 <= head < 4 or type(token) is not int or not 0 <= token < 4:
        raise ValueError('head/token must be integers in [0,3]')
    return head // 2


def rms(row, gain, epsilon, sqrt=math.sqrt):
    denominator = sqrt(sum(v*v for v in row) / len(row) + epsilon)
    return [v * g / denominator for v, g in zip(row, gain)]


def rotate(row, position, theta):
    half = len(row)//2
    out = [0.] * len(row)
    for i in range(half):
        angle = position / theta ** (2*i/len(row))
        c, s = math.cos(angle), math.sin(angle)
        out[i] = row[i]*c - row[i+half]*s
        out[i+half] = row[i+half]*c + row[i]*s
    return out


def softmax(row, exp=math.exp):
    if not row or all(v == -math.inf for v in row):
        raise ValueError('All-masked row: no allowed keys in this teaching reference')
    if any(math.isnan(float(v.detach()) if hasattr(v, 'detach') else v) or v == math.inf for v in row):
        raise ValueError('Invalid softmax input')
    maximum = max(row)
    exps = [exp(v-maximum) for v in row]
    total = sum(exps)
    return [v/total for v in exps]


def compute(f, variant='qwen3', fault='none'):
    validate(f)
    return compute_graph(f, variant, fault)


def compute_graph(f, variant='qwen3', fault='none', *, sqrt=math.sqrt, exp=math.exp, qkv_project=None, output_project=None):
    """Shared GQA graph; differentiable adapter validates tensor shapes before entry."""
    if variant not in ('qwen3', 'qwen25') or fault not in ('none','omit_scale','wrong_group'):
        raise ValueError('Unknown variant/fault')
    x = f['x']
    size = f['dimensions']['S']
    norm = [rms(row, f['input_gain'], f['epsilon'], sqrt) for row in x]
    mixed = qkv_project(norm) if qkv_project else [[sum(norm[t][i]*f['wqkv'][i][j] for i in range(8)) +
              (f['qkv_bias'][j] if variant == 'qwen25' else 0) for j in range(32)] for t in range(size)]
    q, k, v = [], [], []
    for row in mixed:
        q.append([row[g*16+h*4:g*16+h*4+4] for g in range(2) for h in range(2)])
        k.append([row[g*16+8:g*16+12] for g in range(2)])
        v.append([row[g*16+12:g*16+16] for g in range(2)])
    qnorm = [[rms(head, f['q_gain'], f['epsilon'], sqrt) if variant == 'qwen3' else head[:] for head in token] for token in q]
    knorm = [[rms(head, f['k_gain'], f['epsilon'], sqrt) if variant == 'qwen3' else head[:] for head in token] for token in k]
    qr = [[rotate(head, f['positions'][t], f['theta']) for head in token] for t, token in enumerate(qnorm)]
    kr = [[rotate(head, f['positions'][t], f['theta']) for head in token] for t, token in enumerate(knorm)]
    scores, scaled, masked, probs, heads = [], [], [], [], []
    for t in range(size):
        st, sc, ma, pr, ho = [], [], [], [], []
        for h in range(4):
            group = (h//2 + (fault == 'wrong_group')) % 2
            row = [sum(qr[t][h][i]*kr[j][group][i] for i in range(4)) for j in range(size)]
            scale = [n / (1 if fault == 'omit_scale' else 2) for n in row]
            mask = [n if j <= t else -math.inf for j, n in enumerate(scale)]
            p = softmax(mask, exp)
            out = [sum(p[j]*v[j][group][i] for j in range(size)) for i in range(4)]
            st.append(row); sc.append(scale); ma.append(mask); pr.append(p); ho.append(out)
        scores.append(st); scaled.append(sc); masked.append(ma); probs.append(pr); heads.append(ho)
    merged = [[value for head in token for value in head] for token in heads]
    projected = output_project(merged) if output_project else [[sum(merged[t][i]*f['wo'][i][j] for i in range(16))+f['output_bias'][j] for j in range(8)] for t in range(size)]
    residual = [[x[t][j]+projected[t][j] for j in range(8)] for t in range(size)]
    return dict(input=x, norm=norm, mixed=mixed, q=q, k=k, v=v, qnorm=qnorm, knorm=knorm,
                qrope=qr, krope=kr, scores=scores, scaled=scaled, masked=masked,
                probabilities=probs, heads=heads, merged=merged, projected=projected, residual=residual)


def serializable(value):
    if isinstance(value, dict):
        return {k: serializable(v) for k,v in value.items()}
    if isinstance(value, list):
        return [serializable(v) for v in value]
    return '-Infinity' if value == -math.inf else value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write-fixture', action='store_true')
    args = parser.parse_args()
    if args.write_fixture:
        FIXTURE.write_text(json.dumps(authored_fixture(), indent=2)+'\n')
    else:
        f = json.loads(FIXTURE.read_text())
        print(json.dumps({variant: serializable(compute(f, variant)) for variant in ('qwen3','qwen25')}, allow_nan=False))
