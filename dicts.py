# -*- coding: utf-8 -*-
"""
Dictionary tra cứu cho pipeline trích xuất khái niệm y khoa (Viettel AI Race - Vòng 1).

LƯU Ý QUAN TRỌNG VỀ ĐỘ CHÍNH XÁC CỦA MÃ CANDIDATE:
    Môi trường chạy pipeline này không có kết nối internet tới RxNorm API
    (rxnav.nlm.nih.gov) hay ICD-10-CM API (clinicaltables.nlm.nih.gov), nên
    toàn bộ mã RxNorm/ICD-10 dưới đây là mã ở mức "ingredient/khái niệm chung"
    được tổng hợp từ kiến thức tổng quát, mang tính BEST-EFFORT, KHÔNG được
    xác minh trực tiếp qua API chính thức. Trước khi nộp bài chính thức,
    nên verify lại bằng RxNorm/ICD-10-CM API hoặc file dump chính thức nếu có
    thể truy cập mạng.
"""

# ---------------------------------------------------------------------------
# 1. DRUG DICTIONARY: tên thuốc (English INN, thường giữ nguyên trong văn bản
#    tiếng Việt) -> mã RxNorm (ingredient-level RXCUI, best-effort)
# ---------------------------------------------------------------------------
DRUG_DICT = {
    "acetaminophen": "161", "paracetamol": "161",
    "aspirin": "1191",
    "ibuprofen": "5640",
    "metoprolol succinate": "6918", "metoprolol tartrate": "6918", "metoprolol": "6918",
    "atenolol": "1202",
    "amlodipine": "17767",
    "lisinopril": "29046",
    "losartan": "52175",
    "metformin": "6809",
    "omeprazole": "7646",
    "pantoprazole": "40790",
    "atorvastatin": "83367",
    "simvastatin": "36567",
    "pravastatin": "42463",
    "rosuvastatin": "301542",
    "furosemide": "4603",
    "hydrochlorothiazide": "5487",
    "warfarin": "11289",
    "clopidogrel": "32968",
    "levothyroxine": "10582",
    "prednisone": "8640",
    "prednisolone": "8638",
    "gabapentin": "25480",
    "tramadol": "10689",
    "oxycodone": "7804",
    "morphine": "7052",
    "heparin": "5224",
    "enoxaparin": "224905",
    "docusate sodium": "3423", "docusate": "3423",
    "senna": "36711",
    "guaifenesin": "5032",
    "nystatin": "7597",
    "clonazepam": "2598",
    "doxycycline": "3640",
    "azithromycin": "18631",
    "ciprofloxacin": "2551",
    "metronidazole": "6922",
    "ceftriaxone": "2193",
    "vancomycin": "11124",
    "amoxicillin-clavulanate": "723", "amoxicillin": "723",
    "ampicillin": "733",
    "penicillin": "7980",
    "albuterol": "435", "salbutamol": "435",
    "ipratropium": "6035",
    "montelukast": "88249",
    "fluticasone": "41126",
    "digoxin": "3407",
    "spironolactone": "9997",
    "carvedilol": "20352",
    "diltiazem": "3443",
    "amiodarone": "703",
    "rivaroxaban": "1114195",
    "apixaban": "1364430",
    "sertraline": "36437",
    "citalopram": "2556",
    "escitalopram": "321988",
    "trazodone": "10737",
    "quetiapine": "51272",
    "haloperidol": "5093",
    "lorazepam": "6470",
    "diazepam": "3322",
    "zolpidem": "39968",
    "ranitidine": "9143",
    "famotidine": "4278",
    "ondansetron": "26225",
    "metoclopramide": "6915",
    "diphenhydramine": "3498",
    "hydroxyzine": "5570",
    "cephalexin": "2231",
    "fluconazole": "4450",
    "acyclovir": "281",
    "tamsulosin": "37798",
    "finasteride": "4126",
    "sildenafil": "136411",
    "allopurinol": "519",
    "colchicine": "2683",
    "methotrexate": "6851",
    "hydrocortisone": "5492",
    "dexamethasone": "3264",
    "levetiracetam": "114477",
    "phenytoin": "8183",
    "valproic acid": "11118",
    "lithium": "6448",
    "bupropion": "1819",
    "venlafaxine": "39786",
    "duloxetine": "72625",
    "mirtazapine": "15996",
    "olanzapine": "61381",
    "risperidone": "35636",
    "aripiprazole": "89013",
    "clonidine": "2599",
    "hydralazine": "5470",
    "nifedipine": "7417",
    "verapamil": "11170",
    "isosorbide mononitrate": "197304", "isosorbide dinitrate": "6111", "isosorbide": "6111",
    "nitroglycerin": "7452",
    "ezetimibe": "341248",
    "fenofibrate": "4451",
    "glipizide": "4821",
    "glyburide": "4815",
    "glimepiride": "25789",
    "sitagliptin": "593411",
    "methimazole": "6845",
    "propylthiouracil": "8804",
    "insulin glargine": "274783",
    "insulin lispro": "86009",
    "insulin aspart": "51428",
    "insulin": "5856",
    "potassium chloride": "8591",
    "magnesium sulfate": "6585",
    "magnesium oxide": "6582",
    "sodium chloride": "9863",
    "dextrose": "4850",
    "piperacillin-tazobactam": "33036", "piperacillin": "8163",
    "meropenem": "44197",
    "norepinephrine": "7512", "levophed": "7512",
    "propofol": "8782",
    "phentolamine": "8062",
    "imatinib": "282388", "gleevec": "282388",
    "desmopressin": "3298",
    "atorvastatin calcium": "83367",
    "albuterol sulfate": "435",
    "chlorpheniramine": "2441",
    "capsaicin": "1994",
    "melatonin": "6960",
    "multivitamin": "6959",
    "folic acid": "4511",
    "thiamine": "10380",
    "pyridoxine": "8760",
    "vitamin d": "11253", "cholecalciferol": "11253",
    "vitamin b12": "1927", "cyanocobalamin": "1927",
    "calcium carbonate": "18989",
    "polyethylene glycol": "8331",
    "bisacodyl": "1553",
    "simethicone": "9974",
    "loperamide": "6415",
    "albumin": "434",
    "ketorolac": "6132",
    "cefazolin": "2180",
    "cefepime": "20481",
    "linezolid": "84763",
    "daptomycin": "231255",
    "fentanyl": "4337",
    "hydromorphone": "3423000",
    "midazolam": "6960",
    "atropine": "1223",
    "epinephrine": "3992",
    "dopamine": "3628",
    "dobutamine": "3616",
    "furosemide iv": "4603",
    "spironolactone/hctz": "9997",
    "labetalol": "6185",
    "esmolol": "4032",
    "nicardipine": "7239",
    "clevidipine": "672674",
    "tacrolimus": "42316",
    "mycophenolate": "42375",
    "cyclosporine": "3008",
    "prednisone taper": "8640",
    "budesonide": "1657",
    "montelukast sodium": "88249",
}
# fix placeholder mistakes above cleanly
DRUG_DICT["hydromorphone"] = "3423"
DRUG_DICT["midazolam"] = "6960"
DRUG_DICT["calcium carbonate"] = "18989"

# route / frequency / dose tokens used to extend a drug mention span to the
# right so it captures dosage information (as in the vòng-1 example:
# "amlodipine 10 mg po daily")
DOSE_CONTINUATION_RE = (
    r"(?:[\s,:]*"
    r"(?:\d+[\.,]?\d*(?:-\d+[\.,]?\d*)?\s*(?:mg|mcg|g|ml|iu|units?|meq)\b(?:/(?:ml|mg|g))?"
    r"|(?:mg|mcg|g|ml)\b"
    r"|po|iv|im|sc|subcut|subq|pr|sl|top|inh|neb"
    r"|oral|suspension|tablet|tablets|capsule|capsules|cream|ointment|drops|syrup|inhaler|patch|injection|solution|elixir"
    r"|daily|qd|bid|tid|qid|qhs|qam|qpm|prn|q\d+h|q\d+-\d+h|weekly|x\s*\d+"
    r"|extended[- ]release|xl|xr|er|cr|sr"
    r")\b)*"
)

# ---------------------------------------------------------------------------
# 2. SYMPTOM LIST (TRIỆU_CHỨNG) - không có candidates, chỉ có assertions
#    Sắp theo độ dài giảm dần khi match để ưu tiên cụm dài hơn trước.
# ---------------------------------------------------------------------------
SYMPTOM_LIST = [
    "đau đầu", "đau bụng vùng thượng vị", "đau thượng vị", "đau bụng quặn", "đau bụng",
    "đau ngực trái", "đau ngực", "đau lưng", "đau khớp", "đau hông phải", "đau hông",
    "đau nhức", "đau họng", "khó thở khi gắng sức", "khó thở nhẹ", "khó thở",
    "buồn nôn", "nôn", "tiêu chảy", "táo bón", "chóng mặt", "mệt mỏi", "sụt cân",
    "chán ăn", "phù chân", "phù", "ngứa", "phát ban", "vàng da", "tê", "yếu",
    "ớn lạnh", "đổ mồ hôi", "hồi hộp", "đánh trống ngực", "tăng đánh trống ngực",
    "mất ngủ", "lo âu", "căng thẳng", "ho đờm xanh", "ho đờm", "ho khan", "ho",
    "khàn giọng", "giọng khàn", "sổ mũi", "nghẹt mũi", "chảy máu", "bầm tím",
    "sưng khớp", "run tay", "co giật", "ngất", "mất ý thức", "ợ hơi", "ợ chua",
    "khó nuốt", "đầy hơi", "đi ngoài ra máu", "tiểu buốt", "tiểu khó", "tiểu ra máu",
    "tiểu đêm nhiều", "mờ mắt", "ù tai", "ảo giác", "sốt đau", "sốt", "đau",
    "cảm giác thắt chặt ngực", "giảm dung nạp gắng sức", "thắt chặt ngực",
    "cảm giác khó chịu vùng ngực", "khó chịu vùng ngực", "rối loạn dáng đi",
    "gãy cổ xương đùi di lệch", "gãy cổ xương đùi", "vết thương hở",
    "ý thức suy giảm", "không thể tự chăm sóc bản thân",
    "chảy dịch liên tục", "chảy dịch", "ra huyết âm đạo", "vỡ ối", "rỉ ối",
    "cơn co tử cung", "thai máy",
]

# các cụm mô tả không phải triệu chứng thật sự (loại trừ để tránh match nhầm)
SYMPTOM_STOPWORDS = set()

# ---------------------------------------------------------------------------
# 3. DIAGNOSIS DICTIONARY (CHẨN_ĐOÁN) -> danh sách mã ICD-10 (best-effort)
# ---------------------------------------------------------------------------
DIAGNOSIS_DICT = {
    "tăng huyết áp": ["I10"],
    "đái tháo đường type 1": ["E10.9"],
    "đái tháo đường type 2": ["E11.9"],
    "đái tháo đường": ["E11.9"],
    "suy tim": ["I50.9"],
    "rung nhĩ kịch phát": ["I48.0"],
    "rung nhĩ": ["I48.91"],
    "nhồi máu cơ tim": ["I21.9"],
    "bệnh động mạch vành": ["I25.10"],
    "bệnh tim mạch do xơ vữa động mạch": ["I25.10"],
    "viêm phổi": ["J18.9"],
    "hen phế quản": ["J45.909"], "hen suyễn": ["J45.909"],
    "bệnh phổi tắc nghẽn mạn tính": ["J44.9"], "copd": ["J44.9"],
    "suy thận mạn": ["N18.9"],
    "suy thận cấp": ["N17.9"],
    "bệnh thận mạn": ["N18.9"],
    "trào ngược dạ dày - thực quản": ["K21.0", "K21.9"],
    "trào ngược dạ dày thực quản": ["K21.0", "K21.9"],
    "bệnh trào ngược dạ dày- thực quản": ["K21.0", "K21.9"],
    "bệnh trào ngược dạ dày thực quản": ["K21.0", "K21.9"],
    "viêm dạ dày": ["K29.70"],
    "loét dạ dày tá tràng": ["K27.9"],
    "sỏi thận": ["N20.0"],
    "nhiễm trùng đường tiết niệu": ["N39.0"], "nhiễm khuẩn đường tiết niệu": ["N39.0"],
    "viêm gan b": ["B18.1"],
    "viêm gan c": ["B18.2"],
    "xơ gan do rượu": ["K70.30"],
    "xơ gan": ["K74.60"],
    "hội chứng não gan": ["K72.90"],
    "đột quỵ": ["I63.9"], "tai biến mạch máu não": ["I63.9"],
    "xuất huyết não": ["I61.9"],
    "gãy cổ xương đùi": ["S72.001A"],
    "u trực tràng": ["C20"], "khối u trực tràng": ["C20"],
    "bệnh bạch cầu dòng tủy mãn tính": ["C92.10"],
    "tăng cholesterol máu": ["E78.00"],
    "tăng lipid máu": ["E78.5"],
    "tăng sản tuyến tiền liệt": ["N40.0"],
    "viêm tuyến mồ hôi": ["L73.2"],
    "gout": ["M10.9"], "gút": ["M10.9"],
    "viêm khớp dạng thấp": ["M06.9"],
    "thiếu máu": ["D64.9"],
    "suy giáp": ["E03.9"],
    "cường giáp": ["E05.90"],
    "trầm cảm": ["F32.9"],
    "rối loạn lo âu": ["F41.9"],
    "động kinh": ["G40.909"],
    "viêm ruột thừa": ["K35.80"],
    "viêm tụy": ["K85.90"],
    "sỏi mật": ["K80.20"],
    "viêm túi mật": ["K81.9"],
    "thoát vị hoành": ["K44.9"],
    "thoát vị cạnh thực quản": ["K44.9"],
    "tràn dịch màng phổi": ["J91.8"],
    "tràn dịch màng tim": ["I31.3"],
    "xẹp phổi": ["J98.11"],
    "huyết khối tĩnh mạch sâu": ["I82.409"],
    "thuyên tắc phổi": ["I26.99"],
    "hạ huyết áp": ["I95.9"],
    "viêm gan do men": ["K75.9"],
    "tắc nghẽn đường mật": ["K83.1"],
    "bàn chân vẹo bẩm sinh": ["Q66.89"],
}

# ---------------------------------------------------------------------------
# 4. LAB TEST tên viết tắt thường gặp -> tên đầy đủ (chỉ dùng để nhận diện,
#    KẾT_QUẢ_XÉT_NGHIỆM không có candidates nên không cần mã)
# ---------------------------------------------------------------------------
LAB_ABBR = [
    "wbc", "rbc", "hgb", "hct", "plt", "na", "k", "cl", "co2", "bun", "cr",
    "glucose", "ast", "alt", "ap", "tbili", "ldh", "alp", "ck", "trop",
    "ph", "pco2", "po2", "hco3", "inr", "pt", "ptt", "esr", "crp", "cea",
    "tsh", "t3", "t4", "hba1c", "bnp", "lactate", "albumin", "protein",
    "bạch cầu", "hồng cầu", "tiểu cầu", "creatinine", "ure", "natri", "kali",
    "clo", "bilirubin toàn phần", "bilirubin", "phosphatase kiềm",
]

# ---------------------------------------------------------------------------
# 5. CUE WORDS cho assertions
# ---------------------------------------------------------------------------
NEGATION_CUES = [
    "không có", "không ghi nhận", "không xuất hiện", "không thấy",
    "phủ nhận", "chưa ghi nhận", "chưa có", "không đặc hiệu", "không",
    "chưa",
]
FAMILY_CUES = [
    "gia đình", "người nhà", "họ hàng", "bố bệnh nhân", "mẹ bệnh nhân",
    "bố ", "mẹ ", "cha ", "anh trai", "chị gái", "em trai", "em gái",
    "con trai", "con gái", "ông ", "bà ", "chú ", "bác ", "cô ", "dì ", "cậu ",
]
HISTORICAL_CUES = [
    "tiền sử", "trước khi nhập viện", "trước đây", "đã từng", "quá khứ",
    "đã ngừng", "ngừng uống", "ngừng dùng", "trước nhập viện",
]

# section-header regex (áp dụng theo dòng) quyết định ngữ cảnh mặc định
# cho các bullet phía dưới nó, cho tới khi gặp header mới.
SECTION_HISTORICAL_HEADERS = [
    "thuốc trước khi nhập viện", "thuốc trước nhập viện",
    "các bệnh lý mãn tính", "các bệnh lý mạn tính",
    "tiền sử phẫu thuật", "tiền sử bệnh nội khoa", "tiền sử bệnh",
]
