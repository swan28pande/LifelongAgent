"""
Convert HuggingFace nayohan/multi_session_chat to the JSONL format expected by Rsum.

Expected output: data/msc_dialogue/session_{4,5}/test.txt
Each line is a JSON with:
  - previous_dialogs: list of prior sessions, each with a 'dialog' list of {text, id}
  - dialog: current session turns, each {text, id, convai2_id}
  - personas: [[speaker1 personas], [speaker2 personas]]
"""

import json
import os
from collections import defaultdict
from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "msc_dialogue")


def convert_split(hf_split, output_session_ids=(3, 4)):
    """
    Convert a HuggingFace split to per-session JSONL dicts.

    HF session_id 0-4 maps to paper sessions 1-5.
    We output session_ids 3 and 4 (paper sessions 4 and 5).
    """
    # Group by dialoug_id
    by_dial = defaultdict(dict)
    for ex in hf_split:
        did = ex["dialoug_id"]
        sid = ex["session_id"]
        by_dial[did][sid] = ex

    session_data = defaultdict(list)

    for did, sessions in by_dial.items():
        max_sid = max(sessions.keys())

        for target_sid in output_session_ids:
            if target_sid not in sessions:
                continue
            # Build previous_dialogs from sessions 0..(target_sid-1)
            previous_dialogs = []
            for prev_sid in range(target_sid):
                if prev_sid not in sessions:
                    continue
                prev = sessions[prev_sid]
                prev_turns = []
                for i, (text, spk) in enumerate(zip(prev["dialogue"], prev["speaker"])):
                    prev_turns.append({"text": text, "id": spk})
                previous_dialogs.append({"dialog": prev_turns})

            # Build current dialog
            curr = sessions[target_sid]
            curr_turns = []
            for i, (text, spk) in enumerate(zip(curr["dialogue"], curr["speaker"])):
                curr_turns.append({
                    "text": text,
                    "id": spk,
                    "convai2_id": f"{did}-s{target_sid}",
                })

            personas = [
                list(curr["persona1"]),
                list(curr["persona2"]),
            ]

            record = {
                "previous_dialogs": previous_dialogs,
                "dialog": curr_turns,
                "personas": personas,
                "metadata": {
                    "initial_data_id": str(did),
                    "session_id": target_sid + 1,  # 1-indexed for paper
                },
            }
            session_data[target_sid].append(record)

    return session_data


def main():
    print("Loading nayohan/multi_session_chat from HuggingFace...")
    ds = load_dataset("nayohan/multi_session_chat")

    for split_name, hf_split in ds.items():
        print(f"\nProcessing {split_name} split ({len(hf_split)} rows)...")
        session_data = convert_split(hf_split, output_session_ids=(3, 4))

        for sid, records in session_data.items():
            # Paper uses session 4 and 5; HF uses 0-indexed so sid=3→session4, sid=4→session5
            paper_session = sid + 1
            out_dir = os.path.join(DATA_DIR, f"session_{paper_session}")
            os.makedirs(out_dir, exist_ok=True)

            out_file = os.path.join(out_dir, f"{split_name}.txt")
            with open(out_file, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
            print(f"  Session {paper_session} ({split_name}): {len(records)} dialogues → {out_file}")

    # Also create a valid.txt for session_2 (needed by load_example for ICL)
    # Use validation split session 1 (HF sid=1)
    print("\nCreating session_2/valid.txt for ICL examples...")
    hf_val = ds["validation"]
    session_data = convert_split(hf_val, output_session_ids=(1,))
    if 1 in session_data:
        out_dir = os.path.join(DATA_DIR, "session_2")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "valid.txt")
        with open(out_file, "w") as f:
            for rec in session_data[1]:
                f.write(json.dumps(rec) + "\n")
        print(f"  session_2/valid.txt: {len(session_data[1])} records")

    print("\nDone!")


if __name__ == "__main__":
    main()
