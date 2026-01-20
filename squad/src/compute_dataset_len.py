"""Compute the average character length of training sequences.

This script finds all the `training_sequences` fields in a SQuAD dataset JSON 
file and reports the average number of characters in each `training_sequences` 
field.

These numbers were used to create Figures 6 and 7 in the paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Compute the average character length of all strings in the "
			'"training_sequences" fields of a training JSON file.'
		)
	)
	parser.add_argument(
		"input_path",
		type=Path,
		help="Path to the training JSON file (e.g., train_thinking.json)",
	)
	return parser.parse_args()


def iter_training_sequence_lists(data: Any) -> Iterable[list[str]]:
	"""Yield each `training_sequences` list found in the JSON structure."""

	if isinstance(data, dict):
		if "training_sequences" in data and isinstance(data["training_sequences"], list):
			yield data["training_sequences"]
		for value in data.values():
			yield from iter_training_sequence_lists(value)
	elif isinstance(data, list):
		for item in data:
			yield from iter_training_sequence_lists(item)


def main() -> None:
	args = parse_args()

	try:
		raw = args.input_path.read_text(encoding="utf-8")
		data = json.loads(raw)
	except FileNotFoundError:
		raise SystemExit(f"Input file not found: {args.input_path}")
	except json.JSONDecodeError as exc:
		raise SystemExit(f"Failed to parse JSON: {exc}")

	total_chars_all_lists = 0
	total_lists = 0

	for seq_list in iter_training_sequence_lists(data):
		# Sum the characters for the entire list, skipping non-string entries defensively.
		list_char_total = sum(len(seq) for seq in seq_list if isinstance(seq, str))
		total_chars_all_lists += list_char_total
		total_lists += 1

	if total_lists == 0:
		print("No training_sequences lists found.")
		return

	average = total_chars_all_lists / total_lists
	print(
		f"Average characters per training_sequences list: {average:.2f} "
		f"across {total_lists} passages/lists"
	)


if __name__ == "__main__":
	main()
