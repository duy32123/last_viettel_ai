# -*- coding: utf-8 -*-
"""
Pipeline trích xuất + chuẩn hóa khái niệm y khoa cho văn bản tự do tiếng Việt
(Viettel AI Race - Vòng 1 / Bài 2 - Ontological Reasoning in Medical Knowledge
Retrieval).

Cách tiếp cận: rule-based + dictionary lookup (không gọi API/LLM ngoài),
chạy hoàn toàn offline, tuân thủ quy định vòng 1.

Giới hạn đã biết (baseline, không phải giải pháp hoàn chỉnh):
  - Dictionary thuốc/triệu chứng/chẩn đoán/xét nghiệm được xây dựng thủ công,
    độ phủ (recall) giới hạn so với các thuật ngữ y khoa đầy đủ.
  - Mã RxNorm/ICD-10 là best-effort (ingredient-level), chưa verify qua API
    chính thức do môi trường phát triển không có kết nối mạng tới các API đó.
  - Logic xác định assertions (isNegated/isFamily/isHistorical) dựa trên cue
    words + ngữ cảnh section theo dòng, không có mô hình ngữ nghĩa sâu.
"""
import re

from dicts import (
    DRUG_DICT, DOSE_CONTINUATION_RE, SYMPTOM_LIST, DIAGNOSIS_DICT,
    LAB_ABBR, NEGATION_CUES, FAMILY_CUES, HISTORICAL_CUES,
    SECTION_HISTORICAL_HEADERS,
)

# ---------------------------------------------------------------------------
# Build regexes once
# ---------------------------------------------------------------------------


def _alt(keys):
    return "|".join(re.escape(k) for k in sorted(set(keys), key=len, reverse=True))


DRUG_RE = re.compile(r"(" + _alt(DRUG_DICT.keys()) + r")", re.IGNORECASE)
DOSE_RE = re.compile(DOSE_CONTINUATION_RE, re.IGNORECASE)
SYMPTOM_RE = re.compile(r"(" + _alt(SYMPTOM_LIST) + r")", re.IGNORECASE)
DIAG_RE = re.compile(r"(" + _alt(DIAGNOSIS_DICT.keys()) + r")", re.IGNORECASE)

# "<tên xét nghiệm> (chú thích)? là|: <số>" -> bắt cặp TÊN_XÉT_NGHIỆM + KẾT_QUẢ_XÉT_NGHIỆM
LAB_RESULT_RE = re.compile(
    r"(?P<name>[a-zA-ZÀ-ỹà-ỹ][a-zA-ZÀ-ỹà-ỹ0-9 %/\-]{1,40}?(?:\s*\([^)]{1,60}\))?)"
    r"\s*(?:là|:)\s*"
    r"(?P<value>-?\d+[.,]?\d*\s*(?:mg/dl|mmol/l|g/dl|%|u/l|/mm3)?)",
    re.IGNORECASE,
)

HEADER_RE = re.compile(r"^\s*\d+\.\s*(.+)$")

def _compile_cue_res(cues):
    # \b ... \b để tránh match nhầm bên trong 1 từ khác, vd cue "ông " lọt
    # vào giữa từ "không" nếu chỉ so khớp chuỗi con thông thường.
    return [re.compile(r"\b" + re.escape(c.strip()) + r"\b") for c in cues]


NEGATION_CUES_SORTED = _compile_cue_res(sorted(NEGATION_CUES, key=len, reverse=True))
FAMILY_CUES_SORTED = _compile_cue_res(sorted(FAMILY_CUES, key=len, reverse=True))
HISTORICAL_CUES_SORTED = _compile_cue_res(sorted(HISTORICAL_CUES, key=len, reverse=True))


def _find_last_cue_before(cue_res, line_lower, upto):
    """Trả về vị trí (index) của cue xuất hiện gần nhất trước vị trí `upto`
    trong `line_lower`, hoặc -1 nếu không có."""
    best = -1
    segment = line_lower[:upto]
    for cre in cue_res:
        last = None
        for m in cre.finditer(segment):
            last = m
        if last is not None and last.start() > best:
            best = last.start()
    return best


def _line_offsets(text):
    """Trả về list (line_start_offset, line_text) cho từng dòng của text."""
    offsets = []
    start = 0
    for line in text.split("\n"):
        offsets.append((start, line))
        start += len(line) + 1
    return offsets


def _is_header_line(line_stripped):
    """Heuristic: dòng "header/label" thường không bắt đầu bằng '-' (bullet)
    và có xu hướng ngắn hoặc kết thúc bằng ':' (label kèm nội dung)."""
    return not line_stripped.startswith("-")


def extract_concepts(text):
    """Trích xuất danh sách khái niệm y khoa từ 1 văn bản free-text.

    Trả về list các dict theo đúng format output vòng 1 / bài 2.
    """
    results = []
    line_offsets = _line_offsets(text)

    top_section = None      # "history" | "present" | "hospital" | None
    sub_historical = False  # ngữ cảnh "tiền sử" ở cấp sub-header hiện tại

    for line_start, line in line_offsets:
        stripped = line.strip()
        if not stripped:
            continue

        # --- cập nhật ngữ cảnh section ---
        header_match = HEADER_RE.match(line)
        header_text = header_match.group(1) if header_match else None
        low_full = stripped.lower()

        if header_match is not None:
            # top-level numbered section, vd "1.  Tiền sử bệnh nội khoa"
            if "tiền sử" in low_full:
                top_section = "history"
            elif "hiện tại" in low_full:
                top_section = "present"
            else:
                top_section = "hospital"
            sub_historical = "tiền sử" in low_full or any(
                h in low_full for h in SECTION_HISTORICAL_HEADERS
            )
        elif _is_header_line(stripped) and (stripped.endswith(":") or len(stripped) < 60):
            # sub-header dạng nhãn, vd "Thuốc trước khi nhập viện" /
            # "Triệu chứng hiện tại:"
            if any(h in low_full for h in SECTION_HISTORICAL_HEADERS):
                sub_historical = True
            elif "hiện tại" in low_full or "lý do nhập viện" in low_full or "khi nhập viện" in low_full:
                sub_historical = False

        line_lower = line.lower()
        occupied = []  # list (start_local, end_local) đã dùng trong dòng này

        def overlaps(s, e):
            return any(not (e <= a or s >= b) for a, b in occupied)

        # ---------------- THUỐC ----------------
        for m in DRUG_RE.finditer(line):
            s, e = m.start(), m.end()
            if overlaps(s, e):
                continue
            # mở rộng span sang phải để bắt liều/route/tần suất
            dose_m = DOSE_RE.match(line, e)
            if dose_m:
                e = max(e, dose_m.end())
            occupied.append((s, e))
            concept_text = line[s:e].strip()
            # loại bỏ khoảng trắng cuối khỏi span
            trail_ws = len(line[s:e]) - len(line[s:e].rstrip())
            e -= trail_ws
            concept_text = line[s:e]

            drug_key = m.group(1).lower()
            candidates = [DRUG_DICT.get(drug_key, "")]
            candidates = [c for c in candidates if c]

            assertions = _get_assertions(
                line_lower, s, "THUỐC", top_section, sub_historical
            )
            results.append({
                "text": concept_text,
                "type": "THUỐC",
                "candidates": candidates,
                "assertions": assertions,
                "position": [line_start + s, line_start + e],
            })

        # ---------------- CHẨN_ĐOÁN ----------------
        for m in DIAG_RE.finditer(line):
            s, e = m.start(), m.end()
            if overlaps(s, e):
                continue
            occupied.append((s, e))
            diag_key = m.group(1).lower()
            candidates = DIAGNOSIS_DICT.get(diag_key, [])
            assertions = _get_assertions(
                line_lower, s, "CHẨN_ĐOÁN", top_section, sub_historical
            )
            results.append({
                "text": line[s:e],
                "type": "CHẨN_ĐOÁN",
                "candidates": candidates,
                "assertions": assertions,
                "position": [line_start + s, line_start + e],
            })

        # ---------------- XÉT_NGHIỆM (tên + kết quả) ----------------
        for m in LAB_RESULT_RE.finditer(line):
            name = m.group("name").strip()
            name_lower = name.lower()
            if not any(abbr in name_lower for abbr in LAB_ABBR):
                continue
            ns, ne = m.start("name"), m.end("name")
            vs, ve = m.start("value"), m.end("value")
            if overlaps(ns, ne) or overlaps(vs, ve):
                continue
            occupied.append((ns, ne))
            occupied.append((vs, ve))
            results.append({
                "text": line[ns:ne],
                "type": "TÊN_XÉT_NGHIỆM",
                "position": [line_start + ns, line_start + ne],
            })
            results.append({
                "text": line[vs:ve],
                "type": "KẾT_QUẢ_XÉT_NGHIỆM",
                "position": [line_start + vs, line_start + ve],
            })

        # ---------------- TRIỆU_CHỨNG ----------------
        for m in SYMPTOM_RE.finditer(line):
            s, e = m.start(), m.end()
            if overlaps(s, e):
                continue
            occupied.append((s, e))
            assertions = _get_assertions(
                line_lower, s, "TRIỆU_CHỨNG", top_section, sub_historical
            )
            results.append({
                "text": line[s:e],
                "type": "TRIỆU_CHỨNG",
                "assertions": assertions,
                "position": [line_start + s, line_start + e],
            })

    results.sort(key=lambda r: (r["position"][0], r["position"][1]))
    return results


def _get_assertions(line_lower, start_local, ctype, top_section, sub_historical):
    assertions = []

    neg_idx = _find_last_cue_before(NEGATION_CUES_SORTED, line_lower, start_local)
    if neg_idx != -1:
        assertions.append("isNegated")

    fam_idx = _find_last_cue_before(FAMILY_CUES_SORTED, line_lower, start_local)
    if fam_idx != -1:
        assertions.append("isFamily")

    # isHistorical chỉ áp dụng cho THUỐC và CHẨN_ĐOÁN theo mặc định (khớp với
    # ví dụ chính thức của vòng 1: các TRIỆU_CHỨNG mô tả lý do dùng thuốc
    # không tự động bị gắn isHistorical trừ khi có cue rõ ràng ngay trong câu
    # mô tả bản thân triệu chứng, vd "có tiền sử ho ra máu").
    is_hist = False
    if ctype in ("THUỐC", "CHẨN_ĐOÁN"):
        hist_idx = _find_last_cue_before(HISTORICAL_CUES_SORTED, line_lower, start_local)
        if hist_idx != -1:
            is_hist = True
        if sub_historical:
            is_hist = True
    elif ctype == "TRIỆU_CHỨNG":
        # chỉ gắn isHistorical cho triệu chứng khi cue "tiền sử" nằm ngay
        # sát trước (trong khoảng ký tự ngắn) để tránh lan rộng sai
        idx = line_lower.rfind("tiền sử", 0, start_local)
        if idx != -1 and start_local - idx < 20:
            is_hist = True
    if is_hist:
        assertions.append("isHistorical")

    # loại trùng, giữ thứ tự ưu tiên
    seen = []
    for a in assertions:
        if a not in seen:
            seen.append(a)
    return seen[:3]
