from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
COT_DIR = DATA_DIR / "cot"
EVAL_DIR = DATA_DIR / "eval"

for directory in [RAW_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, COT_DIR, EVAL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Models
SUMMARY_MODEL_NAME = "IlyaGusev/rut5_base_sum_gazeta"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
LLM_MODEL_NAME = "deepseek/deepseek-r1-0528:free"

# Raw data
RAW_TSV_FILE = RAW_DIR / "RuMedPrimeData.tsv"
ICD10_DICT_FILE = RAW_DIR / "icd10_ru.json"

# Main files
NORMALIZED_CASES_FILE = PROCESSED_DIR / "RuMedPrimeData_normalized_full.json"
PROCESSED_LLM_FILE = PROCESSED_DIR / "RuMedPrimeData_processed_llm.json"
FINAL_PROCESSED_FILE = PROCESSED_DIR / "RuMedPrimeData_processed_final.json"

EMBEDDINGS_NPY_FILE = EMBEDDINGS_DIR / "RuMedPrimeData_embeddings.npy"
EMBEDDINGS_META_FILE = EMBEDDINGS_DIR / "RuMedPrimeData_embeddings_meta.json"

EMBEDDINGS_FINAL_NPY_FILE = EMBEDDINGS_DIR / "RuMedPrimeData_embeddings_final.npy"
EMBEDDINGS_FINAL_META_FILE = EMBEDDINGS_DIR / "RuMedPrimeData_embeddings_final_meta.json"

# CoT
COT_CASES_FILE = COT_DIR / "cot_cases.json"
COT_EMBEDDINGS_NPY_FILE = COT_DIR / "cot_embeddings.npy"
COT_EMBEDDINGS_META_FILE = COT_DIR / "cot_embeddings_meta.json"

# Evaluation
EVAL_TEST_CASES_FILE = EVAL_DIR / "test_cases.json"
EVAL_RESULTS_FILE = EVAL_DIR / "eval_results.json"

ICD10_PDF_FILE = RAW_DIR / "rycm7ud7ylv1ah5m15z0w0dgx5sv3hcr.pdf"
ICD10_DICT_FILE = RAW_DIR / "icd10_ru.json"

# Params
TOP_K = 5
N_RUNS = 5
SAVE_EVERY = 25

MAX_INPUT_LEN = 600
MAX_OUTPUT_LEN = 128

NO_REPEAT_NGRAM_SIZE = 4
NUM_BEAMS = 4
EARLY_STOPPING = True

DEFAULT_CONSENSUS_METHOD = "hybrid"
COT_TOP_K = 3
DIALOGUE_MAX_TURNS = 7
DIALOGUE_CONFIDENCE_THRESHOLD = 0.75