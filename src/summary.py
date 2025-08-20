from __future__ import annotations

import torch

from src.eval_lstm import build_prefix_and_suffix, rouge_1_2_metrics


def format_score(value):
	"""Format a numeric metric as a fixed-precision string; return 'N/A' for NaN.

	This utility keeps printing concise and robust across missing/NaN values.
	"""
	try:
		if value is None:
			return "N/A"
		if isinstance(value, float) and value != value:
			return "N/A"
		return f"{float(value):.3f}"
	except Exception:
		return "N/A"


def print_scores(label, scores):
	"""Pretty-print a dict of ROUGE metrics with a section label."""
	print(f"\n{label}")
	print(
		{
			"rouge1_p": format_score(scores.get("rouge1_p")),
			"rouge1_r": format_score(scores.get("rouge1_r")),
			"rouge1_f1": format_score(scores.get("rouge1_f1")),
			"rouge2_p": format_score(scores.get("rouge2_p")),
			"rouge2_r": format_score(scores.get("rouge2_r")),
			"rouge2_f1": format_score(scores.get("rouge2_f1")),
		}
	)


def print_comparison(base_label, base, other_label, other):
	"""Print per-metric deltas between two result dicts (other - base)."""
	def delta(key: str) -> str:
		b = base.get(key)
		o = other.get(key)
		if b is None or o is None:
			return "N/A"
		if isinstance(b, float) and b != b:
			return "N/A"
		if isinstance(o, float) and o != o:
			return "N/A"
		return f"{(o - b):+.3f}"

	print(f"\nDelta {other_label} vs {base_label} (positive means {other_label} better):")
	print(
		{
			"rouge1_f1": delta("rouge1_f1"),
			"rouge2_f1": delta("rouge2_f1"),
			"rouge1_p": delta("rouge1_p"),
			"rouge1_r": delta("rouge1_r"),
			"rouge2_p": delta("rouge2_p"),
			"rouge2_r": delta("rouge2_r"),
		}
	)


def show_examples(model, texts, vocab, device, max_new_tokens=20, k=8):
	"""Display qualitative prefix→prediction vs reference comparisons.

	The function follows the assignment rule (75% prefix, 25% reference tail),
	and prints the decoded strings for quick manual inspection.
	"""
	model.eval()
	shown = 0
	for t in texts:
		if shown >= k:
			break
		ids = vocab.encode(t, add_special_tokens=True)
		prefix, suffix = build_prefix_and_suffix(ids, vocab.bos_id, vocab.eos_id)
		if not suffix:
			continue
		ref_tail = suffix[:max_new_tokens]
		gen = model.generate(
			torch.tensor(prefix, dtype=torch.long, device=device),
			max_new_tokens=max_new_tokens,
			eos_token_id=vocab.eos_id,
		)
		pred_tail = gen[0].tolist()[len(prefix) :]
		print("\n---")
		print("prefix:", vocab.decode(prefix))
		print("prefix_len:", len(prefix), "ref_tail_len:", len(ref_tail))
		print("pred  :", vocab.decode(pred_tail))
		print("ref   :", vocab.decode(ref_tail))
		print(
			"ROUGE1-F1:",
			round(
				rouge_1_2_metrics(
					pred_tail, ref_tail, vocab.pad_id, vocab.bos_id, vocab.eos_id
				)["rouge1_f1"],
				4,
			),
		)
		shown += 1


__all__ = [
	"format_score",
	"print_scores",
	"print_comparison",
	"show_examples",
]


