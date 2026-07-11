import json
import sys
from pathlib import Path
from schemas import SUPPORTED_CARD_TYPES


def convert_ls_export(ls_export_path: str, output_jsonl: str, card_type: str):
    """
    Convert a Label Studio JSON export to Donut metadata.jsonl.

    Args:
        ls_export_path: path to the Label Studio export JSON file
        output_jsonl:   destination metadata.jsonl path
        card_type:      one of SUPPORTED_CARD_TYPES — written as the card_type field
    """
    if card_type not in SUPPORTED_CARD_TYPES:
        print(f"Unknown card_type '{card_type}'. Supported: {SUPPORTED_CARD_TYPES}")
        sys.exit(1)

    with open(ls_export_path) as f:
        annotations = json.load(f)

    records = []
    for item in annotations:
        # Label Studio prepends an 8-char hash + hyphen to uploaded filenames.
        raw_file_name = Path(item["data"]["image"]).name
        file_name = raw_file_name.split("-", 1)[1] if "-" in raw_file_name else raw_file_name

        result = item["annotations"][0]["result"]

        ground_truth = {"card_type": card_type}
        for field in result:
            key = field["from_name"]
            value = field["value"]["text"][0] if field["value"]["text"] else ""
            ground_truth[key] = value.strip()

        records.append({"file_name": file_name, "ground_truth": ground_truth})

    with open(output_jsonl, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Converted {len(records)} annotations → {output_jsonl}  (card_type: {card_type})")


if __name__ == "__main__":
    # Usage: python convert_labelstudio.py <ls_export.json> <output.jsonl> <card_type>
    # Examples:
    #   python src/convert_labelstudio.py data/real/label_studio_export.json       data/real/metadata.jsonl            indian_passport
    #   python src/convert_labelstudio.py data/are_fed_card/ls_export.json         data/are_fed_card/metadata.jsonl    are_fed_card
    #   python src/convert_labelstudio.py data/cod/ls_export.json                  data/cod/metadata.jsonl             cod_passport
    #   python src/convert_labelstudio.py data/zwe/ls_export.json                  data/zwe/metadata.jsonl             zwe_passport
    if len(sys.argv) != 4:
        print("Usage: python src/convert_labelstudio.py <ls_export.json> <output.jsonl> <card_type>")
        print(f"Supported card types: {SUPPORTED_CARD_TYPES}")
        sys.exit(1)

    convert_ls_export(
        ls_export_path=sys.argv[1],
        output_jsonl=sys.argv[2],
        card_type=sys.argv[3],
    )
