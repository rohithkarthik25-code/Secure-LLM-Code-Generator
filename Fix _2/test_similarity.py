import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "capability_2", "capextract_1_new_bay")))
from capextract.core.vector_mapper import VectorMapper
from sentence_transformers import util
import torch

mapper = VectorMapper.get_instance()
print("Testing similarity clustering on CodeBERT...")
test_text = "bubble_sort(arr)"
func_emb = mapper.model.encode(test_text, convert_to_tensor=True)

scores = []
for cap, anchor_embs in mapper.anchor_embeddings.items():
    cos_scores = util.cos_sim(func_emb, anchor_embs)[0]
    max_score = torch.max(cos_scores).item()
    scores.append((cap.value, max_score))

scores.sort(key=lambda x: x[1], reverse=True)
print("\nTop 10 matches:")
for name, val in scores[:10]:
    print(f"  {name}: {val:.4f}")

best_score = scores[0][1]
for threshold_drop in [0.10, 0.05, 0.02, 0.01, 0.005]:
    optimal_matches = [m for m in scores if (best_score - m[1]) <= threshold_drop]
    print(f"Drop-off threshold {threshold_drop:.3f}: matched {len(optimal_matches)} primitives")

