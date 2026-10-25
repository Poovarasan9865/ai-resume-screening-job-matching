# AI Resume Screening & Job Matching

A beginner-friendly portfolio project that compares a PDF resume with a job description using semantic embeddings and skill overlap.

## Features

- Upload a text-based PDF resume.
- Paste a job description.
- Semantic similarity using `sentence-transformers/all-MiniLM-L6-v2`.
- Detect skills from a small editable skill dictionary.
- Show matched skills, detected missing skills, and additional resume skills.
- FastAPI backend and Streamlit user interface.
- Download the result as JSON.

> Important: the scores are approximate relevance indicators, not hiring probabilities or decisions. The skill list is intentionally basic and should be expanded and reviewed. Use job-related criteria and human review; never use protected characteristics to rank or reject people.

## Folder structure

```text
ai_resume_screening_job_matching/
├── app.py
├── main.py
├── resume_analyzer.py
├── requirements.txt
├── sample_job_description.txt
├── README.md
└── .gitignore
```

## Windows setup (PowerShell)

Open PowerShell in this folder.

### 1. Create and activate a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this for the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. Start the FastAPI backend

Terminal 1:

```powershell
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/health` to check the API and `http://127.0.0.1:8000/docs` to test endpoints.

### 3. Start the Streamlit interface

Open a second terminal in the same folder, activate the environment, then run:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Open the local URL Streamlit prints, usually `http://localhost:8501`.

The first analysis downloads the embedding model, so it can take time and needs internet access. Later runs should reuse the cached model.

## API endpoint

`POST /analyze` accepts:
- `resume`: PDF file
- `job_description`: text form field

`POST /analyze-text` accepts JSON:

```json
{
  "resume_text": "Python, SQL, machine learning and FastAPI project experience.",
  "job_description": "Looking for a Python engineer with SQL and machine learning skills."
}
```

## How the score works

1. Extract text from the PDF.
2. Encode resume and job description with the same sentence-transformer model.
3. Compute cosine similarity between the two vectors.
4. Detect known skills and compare the sets.
5. Return semantic similarity, skill coverage, matched skills, and missing skills.

Semantic similarity may be affected by resume length and wording. It should not be treated as an objective measure of candidate quality.

## Common problems

- **`No module named ...`**: activate `.venv` and run `pip install -r requirements.txt`.
- **API connection error**: ensure `uvicorn main:app --reload` is running in terminal 1.
- **Scanned PDF has no text**: use a text-based PDF or add an OCR step.
- **First request is slow**: the sentence-transformer model may be downloading.
- **Memory/installation issues**: use Python 3.11 or 3.12 in a fresh virtual environment if a dependency does not support your Python version.

## Next improvements

- OCR support for scanned resumes.
- Add a database to save consented test records.
- Add more skills and aliases to `SKILL_BANK`.
- Add authentication, upload limits, logging, tests, and deployment configuration.
- Add multilingual resume support and evaluate against a manually reviewed test set.
