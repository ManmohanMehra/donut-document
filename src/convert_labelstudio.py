import json
from pathlib import Path

def convert_ls_export(ls_export_path: str, output_jsonl: str):
    with open(ls_export_path) as f:
        annotations = json.load(f)

    records = []
    for item in annotations:
        # Get filename like '1e1856de-pass_2.png'
        raw_file_name = Path(item["data"]["image"]).name
        
        # Label studio adds an 8-character hash and hyphen to uploaded files.
        # We split by the first hyphen to get the original filename 'pass_2.png'
        if "-" in raw_file_name:
            file_name = raw_file_name.split("-", 1)[1]
        else:
            file_name = raw_file_name
            
        result = item["annotations"][0]["result"]

        ground_truth = {"card_type": "indian_passport"}
        for field in result:
            key = field["from_name"]
            value = field["value"]["text"][0] if field["value"]["text"] else ""
            ground_truth[key] = value.strip()

        records.append({"file_name": file_name, "ground_truth": ground_truth})

    with open(output_jsonl, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Converted {len(records)} annotations → {output_jsonl}")

if __name__ == "__main__":
    convert_ls_export(
        "data/real/label_studio_export.json",
        "data/real/metadata.jsonl"
    )
