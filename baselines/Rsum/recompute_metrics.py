"""Recompute metrics using the paper's exact evaluation formula."""
import json, re, collections
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


def normalize(s):
    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    return ' '.join(s.split())


def paper_f1(preds, refs):
    total = 0
    for p, r in zip(preds, refs):
        p_toks = word_tokenize(normalize(p))
        r_toks = word_tokenize(normalize(r))
        common = collections.Counter(p_toks) & collections.Counter(r_toks)
        n = sum(common.values())
        if not p_toks or not r_toks or n == 0:
            continue
        prec, rec = n / len(p_toks), n / len(r_toks)
        total += 2 * prec * rec / (prec + rec)
    return total / len(preds) * 100


def paper_bleu(predictions, labels):
    # Exact replication of utils/evaluation.py compute_bleu
    inf_tok = [word_tokenize(i) for i in labels]
    tra_tok = [word_tokenize(t) for t in predictions]
    chencherry = SmoothingFunction()
    weights = [(0.5, 0.5), (0.333, 0.333, 0.334)]
    scores = corpus_bleu(inf_tok, tra_tok, weights=weights,
                         smoothing_function=chencherry.method7)
    return [s * 100 for s in scores]


PAPER = {
    "ChatGPT-Rsum (Ours)":    {"F1": 20.48, "BLEU1": 21.83, "BLEU2": 12.59},
    "ChatGPT (context-only)": {"F1": 19.41, "BLEU1": 21.23, "BLEU2": 12.24},
}

for name, fname in [("ChatGPT-Rsum (Ours)", "results/rsum_predictions.json"),
                    ("ChatGPT (context-only)", "results/context_only_predictions.json")]:
    try:
        data = json.load(open(fname))
        preds, refs = data["predictions"], data["references"]
        f1 = paper_f1(preds, refs)
        bleu = paper_bleu(preds, refs)
        p = PAPER[name]
        print(f"\n{'='*55}")
        print(f"  {name}  (n={len(preds)})")
        print(f"{'='*55}")
        print(f"  F1:     {f1:5.2f}   paper={p['F1']}   diff={f1-p['F1']:+.2f}")
        print(f"  BLEU-1: {bleu[0]:5.2f}   paper={p['BLEU1']}  diff={bleu[0]-p['BLEU1']:+.2f}")
        print(f"  BLEU-2: {bleu[1]:5.2f}   paper={p['BLEU2']}  diff={bleu[1]-p['BLEU2']:+.2f}")
    except FileNotFoundError:
        print(f"\n{name}: predictions not found at {fname}")
