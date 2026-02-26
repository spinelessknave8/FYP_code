import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


REPLACEMENTS = {
    "I adapt quickly to change.": "I adjust quickly when priorities change.",
    "I advocate strongly for sound decisions.": "I champion decisions grounded in evidence.",
    "I cultivate teamwork and trust.": "I build team trust through consistent collaboration.",
    "I ensure accuracy and attention to detail.": "I safeguard quality through careful detail checks.",
    "I extract insights from data.": "I turn data into practical insights.",
    "I follow rules for consistency.": "I apply standards consistently to reduce variance.",
    "I handle pressure well.": "I perform steadily under pressure.",
    "I handle time pressure effectively.": "I manage tight deadlines without losing quality.",
    "I manage pressure calmly.": "I stay composed and effective under pressure.",
    "I persevere through difficulties.": "I keep moving forward through setbacks.",
    "I persist despite obstacles.": "I continue delivering despite obstacles.",
    "I present recommendations convincingly.": "I communicate recommendations in a persuasive, structured way.",
    "I remain calm when stakes are high.": "I stay level-headed in high-stakes situations.",
    "I respect policies and standards.": "I follow policies and standards rigorously.",
    "I set direction and clarify outcomes.": "I define direction and make outcomes explicit.",
    "I standardize for reliability.": "I standardize workflows to improve reliability.",
    "I standardize processes to ensure quality.": "I formalize processes to maintain quality.",
    "I stay calm under pressure.": "I remain calm and focused under pressure.",
    "I test assumptions with evidence.": "I validate assumptions with objective evidence.",
}


def norm(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def duplicate_groups(path: Path) -> int:
    occ = defaultdict(int)
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = json.loads(row["content"])
            for option in content.get("options", []):
                occ[norm(option.get("text", ""))] += 1
    return sum(1 for count in occ.values() if count > 1)


def main(path: Path, backup_path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if fieldnames is None:
        raise ValueError("CSV appears to have no header row")

    if not backup_path.exists():
        backup_path.write_text(path.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")

    seen = defaultdict(int)
    changed = 0
    for row in rows:
        content = json.loads(row["content"])
        for option in content.get("options", []):
            text = option.get("text", "")
            if text in REPLACEMENTS:
                seen[text] += 1
                if seen[text] > 1:
                    option["text"] = REPLACEMENTS[text]
                    changed += 1
        row["content"] = json.dumps(content, ensure_ascii=False, separators=(",", ":"))

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated option entries: {changed}")
    print(f"Remaining duplicate option-text groups: {duplicate_groups(path)}")
    print(f"Backup: {backup_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to Supabase Snippet TalentQ Questions.csv")
    parser.add_argument(
        "--backup",
        default="Supabase Snippet TalentQ Questions.backup.csv",
        help="Backup CSV path (created once).",
    )
    args = parser.parse_args()
    main(Path(args.csv), Path(args.backup))
