# -*- coding: utf-8 -*-
"""
Chạy pipeline trích xuất trên toàn bộ input/*.txt và sinh output/*.json
theo đúng format yêu cầu của Vòng 1 - Viettel AI Race.

Cách dùng:
    python3 run_pipeline.py <input_dir> <output_dir>

Ví dụ:
    python3 run_pipeline.py ./test/input ./output
"""
import json
import os
import sys

from extract import extract_concepts


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 run_pipeline.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir, output_dir = sys.argv[1], sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    files.sort(key=lambda f: int(os.path.splitext(f)[0]))

    for fname in files:
        record_id = os.path.splitext(fname)[0]
        with open(os.path.join(input_dir, fname), encoding="utf-8") as fh:
            text = fh.read()

        concepts = extract_concepts(text)

        out_path = os.path.join(output_dir, f"{record_id}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(concepts, fh, ensure_ascii=False, indent=2)

    print(f"Đã xử lý {len(files)} file. Kết quả tại: {output_dir}")


if __name__ == "__main__":
    main()
