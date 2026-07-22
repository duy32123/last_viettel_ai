from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.models.ner.labels import TARGET_TYPES

ASSERTION_TYPES = {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"}
ALLOWED_ASSERTIONS = ("isNegated", "isFamily", "isHistorical")
TYPE_LOOKUP = {value.casefold(): value for value in TARGET_TYPES}
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
_GARBAGE_RE = re.compile(
    r"^(?:n/?a|bid|tid|qid|qd|prn|po|iv|im|sc|\*+|[-–—.,:;/%]|\d)$",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """Bạn là bộ trích xuất khái niệm y khoa tiếng Việt.
Chỉ trả về một JSON array. Mỗi phần tử có:
- text: chuỗi con NGUYÊN VĂN, không sửa chính tả hay diễn giải;
- type: đúng một trong TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM,
  KẾT_QUẢ_XÉT_NGHIỆM, CHẨN_ĐOÁN, THUỐC;
- start, end: vị trí ký tự 0-based, end-exclusive trong đoạn được cung cấp;
- assertions: chỉ cho TRIỆU_CHỨNG/CHẨN_ĐOÁN/THUỐC, là danh sách gồm
  isNegated, isFamily, isHistorical khi thực sự phù hợp.

Liệt kê từng lần xuất hiện theo thứ tự. Không sinh candidates hay mã ICD/RxNorm.
Không trích N/A, dấu che *****, đơn vị hoặc ký tự rời. Không thêm markdown.
Văn bản có thể là bệnh án hoặc bài giáo dục sức khỏe; chỉ trích khái niệm thực sự
được nhắc trong văn bản, không suy diễn bệnh của bệnh nhân."""


def build_messages(text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"VĂN BẢN (độ dài {len(text)} ký tự):\n{text}"},
    ]


def _salvage_array(text: str) -> list[Any]:
    if not text.lstrip().startswith("["):
        raise json.JSONDecodeError("response is not a JSON array", text, 0)
    decoder = json.JSONDecoder()
    rows: list[Any] = []
    cursor = text.find("[") + 1
    while cursor < len(text):
        while cursor < len(text) and text[cursor] in " \t\r\n,":
            cursor += 1
        if cursor >= len(text) or text[cursor] == "]":
            break
        try:
            value, cursor = decoder.raw_decode(text, cursor)
        except json.JSONDecodeError:
            break
        rows.append(value)
    if not rows:
        raise json.JSONDecodeError("no complete object in truncated array", text, cursor)
    return rows


def parse_json_response(raw: str) -> list[dict[str, Any]]:
    text = raw.strip()
    fence = _FENCE_RE.match(text)
    if fence:
        text = fence.group(1).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = _salvage_array(text)
    if isinstance(value, dict):
        value = value.get("entities", value.get("concepts"))
    if not isinstance(value, list):
        raise ValueError("LLM response must contain a JSON entity array")
    return [row for row in value if isinstance(row, dict)]


def _canonical_type(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return TYPE_LOOKUP.get(value.strip().casefold())


def _find_occurrence(
    source: str,
    surface: str,
    hint: int | None,
    used: set[tuple[int, int]],
    cursor: dict[str, int],
) -> tuple[int, int] | None:
    matches = [(m.start(), m.end()) for m in re.finditer(re.escape(surface), source)]
    if not matches:
        return None
    available = [span for span in matches if span not in used]
    if not available:
        return None
    if hint is not None:
        return min(available, key=lambda span: (abs(span[0] - hint), span[0]))
    key = surface
    start_at = cursor.get(key, 0)
    later = [span for span in available if span[0] >= start_at]
    chosen = (later or available)[0]
    cursor[key] = chosen[1]
    return chosen


def align_predictions(source: str, raw_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Align untrusted LLM rows to exact source substrings.

    Model offsets are accepted only when they reproduce ``text`` exactly. Otherwise
    they are merely a hint used to choose among repeated literal occurrences.
    """
    aligned: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    used: set[tuple[int, int]] = set()
    cursor: dict[str, int] = {}
    for raw in raw_rows:
        surface = raw.get("text")
        typ = _canonical_type(raw.get("type"))
        if not isinstance(surface, str) or not surface or typ is None:
            dropped.append({"row": raw, "reason": "invalid_text_or_type"})
            continue
        if _GARBAGE_RE.fullmatch(surface.strip()) or len(surface.strip()) <= 1:
            dropped.append({"row": raw, "reason": "garbage_surface"})
            continue
        start = raw.get("start")
        end = raw.get("end")
        exact = (
            isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start < end <= len(source)
            and source[start:end] == surface
            and (start, end) not in used
        )
        span = (start, end) if exact else _find_occurrence(
            source, surface, start if isinstance(start, int) else None, used, cursor
        )
        if span is None:
            dropped.append({"row": raw, "reason": "no_exact_substring"})
            continue
        start, end = span
        row: dict[str, Any] = {
            "text": source[start:end],
            "type": typ,
            "start": start,
            "end": end,
            # LLM confidence is not calibrated and must not control overlap removal.
            "score": 0.8,
        }
        if typ in ASSERTION_TYPES:
            supplied = raw.get("assertions", [])
            if not isinstance(supplied, list):
                supplied = []
            row["assertions"] = [name for name in ALLOWED_ASSERTIONS if name in supplied]
        aligned.append(row)
        used.add(span)
    return aligned, dropped


def _remove_overlaps(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    # BTC annotations are flat. Prefer confidence, then the more informative span.
    ranked = sorted(
        rows,
        key=lambda row: (-float(row.get("score", 0.0)), -(row["end"] - row["start"]), row["start"], row["type"]),
    )
    kept: list[dict[str, Any]] = []
    for row in ranked:
        if any(not (row["end"] <= old["start"] or row["start"] >= old["end"]) for old in kept):
            continue
        kept.append(row)
    kept.sort(key=lambda row: (row["start"], row["end"], row["type"]))
    return kept, len(rows) - len(kept)


def _chunks(text: str, max_chars: int, overlap: int) -> list[tuple[int, str]]:
    if max_chars <= 0 or len(text) <= max_chars:
        return [(0, text)]
    overlap = max(0, min(overlap, max_chars // 2))
    chunks: list[tuple[int, str]] = []
    start = 0
    while start < len(text):
        hard_end = min(len(text), start + max_chars)
        end = hard_end
        if hard_end < len(text):
            boundary = max(text.rfind("\n", start + max_chars // 2, hard_end), text.rfind(". ", start + max_chars // 2, hard_end))
            if boundary > start:
                end = boundary + 1
        chunks.append((start, text[start:end]))
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


@dataclass
class TransformersGenerator:
    model_path: str
    adapter_path: str | None = None
    device: str = "auto"
    max_new_tokens: int = 4096

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        local = Path(self.model_path).exists()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True, local_files_only=local)
        kwargs: dict[str, Any] = {"local_files_only": local, "torch_dtype": "auto"}
        if self.device == "auto":
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
        if self.adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.adapter_path, is_trainable=False)
        if self.device != "auto":
            self.model.to(self.device)
        self.model.eval()
        self._torch = torch

    def __call__(self, messages: list[dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with self._torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


class LLMHybridExtractor:
    def __init__(
        self,
        generator: Callable[[list[dict[str, str]]], str],
        *,
        max_chunk_chars: int = 12000,
        chunk_overlap_chars: int = 512,
    ) -> None:
        self.generator = generator
        self.max_chunk_chars = max_chunk_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.generation_calls = 0
        self.chunk_count = 0
        self.dropped_rows = 0
        self.dropped_overlaps = 0

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "LLMHybridExtractor":
        generator = TransformersGenerator(
            model_path=cfg["model_path"],
            adapter_path=cfg.get("adapter_path"),
            device=cfg.get("device", "auto"),
            max_new_tokens=int(cfg.get("max_new_tokens", 4096)),
        )
        return cls(
            generator,
            max_chunk_chars=int(cfg.get("max_chunk_chars", 12000)),
            chunk_overlap_chars=int(cfg.get("chunk_overlap_chars", 512)),
        )

    def predict(self, text: str) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        for base, chunk in _chunks(text, self.max_chunk_chars, self.chunk_overlap_chars):
            self.chunk_count += 1
            raw = self.generator(build_messages(chunk))
            self.generation_calls += 1
            parsed = parse_json_response(raw)
            aligned, dropped = align_predictions(chunk, parsed)
            self.dropped_rows += len(dropped)
            for row in aligned:
                row = dict(row)
                row["start"] += base
                row["end"] += base
                row["text"] = text[row["start"] : row["end"]]
                collected.append(row)
        # First collapse duplicate chunk predictions, then remove nested/partial spans.
        best: dict[tuple[int, int, str], dict[str, Any]] = {}
        for row in collected:
            key = (row["start"], row["end"], row["type"])
            if key not in best or row.get("score", 0.0) > best[key].get("score", 0.0):
                best[key] = row
        rows, overlap_count = _remove_overlaps(list(best.values()))
        self.dropped_overlaps += overlap_count + len(collected) - len(best)
        return rows