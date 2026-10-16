"""Spike: which browser gate threshold (Xenova fp32 L-12, 512 tokens) best reproduces the server's flashrank
gate at 0.65 on the golden set? Uses the pairs parity.py exported to demo/spike/.out/rerank_reference.json."""
import json
import sys

import numpy as np

sys.path.insert(0, "demo/spike")
from parity import RERANKER, RERANKER_REV, onnx_rerank, onnx_session, tokenizer   # noqa: E402

ref = json.load(open("demo/spike/.out/rerank_reference.json"))
golden = json.load(open("data/eval/golden_dataset.json"))
in_scope = [bool(g.get("in_scope")) for g in golden]
sess, tok = onnx_session(RERANKER, RERANKER_REV, "onnx/model.onnx"), tokenizer(RERANKER, RERANKER_REV, 512)
flash = np.array([max(r["flashrank_scores"]) for r in ref])
fp32 = np.array([float(onnx_rerank(sess, tok, r["query"], r["passages"]).max()) for r in ref])
print("pair check: first pair fp32 sigmoid", float(onnx_rerank(sess, tok, "What port must be open for SSL?",
      ["The router TX-9000 requires port 443 open for SSL connections."])[0]))
server = flash >= 0.65
best = None
for t in np.round(np.arange(0.50, 0.995, 0.01), 2):
    browser = fp32 >= t
    agree = int((browser == server).sum())
    trick = int(((browser == server) & ~np.array(in_scope)).sum())
    answered_in = int((browser & np.array(in_scope)).sum())
    if best is None or agree > best[1]:
        best = (t, agree, trick, answered_in)
    if t in (0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99):
        print(f"t={t:.2f} agree={agree}/67 trick_agree={trick}/20 in_scope_answered={answered_in}/47")
print("server answers in scope:", int((server & np.array(in_scope)).sum()), "/47; tricks passed:", int((server & ~np.array(in_scope)).sum()))
print("best threshold", best)
print("trick question top scores (fp32):", sorted(np.round(fp32[~np.array(in_scope)], 3).tolist(), reverse=True)[:5])
print("disagreements at 0.65:", [(golden[i]["question"][:60], round(flash[i], 3), round(fp32[i], 3))
                                 for i in range(len(ref)) if (fp32[i] >= 0.65) != server[i]])
