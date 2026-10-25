
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from resume_analyzer import analyze_resume, extract_pdf_text


app = FastAPI(
    title="AI Resume Screening & Job Matching API",
    description="Analyze resumes against job descriptions using embeddings and an LLM.",
    version="1.0.0"
)


class TextAnalysisRequest(BaseModel):
    resume_text: str = Field(
        min_length=1,
        description="Plain-text resume"
    )
    job_description: str = Field(
        min_length=1,
        description="Job description"
    )


@app.get("/")
def root():
    return {
        "message": "AI Resume Screening & Job Matching API is running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "AI Resume Screening API is healthy"
    }

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    favicon_path = Path(__file__).parent / "static" / "favicon.ico"

    if favicon_path.is_file():
        return FileResponse(favicon_path)

    raise HTTPException(
        status_code=404,
        detail="Favicon not configured"
    )


@app.post("/analyze")
async def analyze_pdf(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    if not resume.filename or not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF resume."
        )

    if not job_description.strip():
        raise HTTPException(
            status_code=422,
            detail="Job description cannot be empty."
        )

    try:
        content = await resume.read()

        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="PDF must be 10 MB or smaller."
            )

        if not content:
            raise HTTPException(
                status_code=422,
                detail="The uploaded PDF is empty."
            )

        resume_text = extract_pdf_text(content)

        result = analyze_resume(
            resume_text,
            job_description
        )

        result["resume_filename"] = resume.filename

        return result

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        print(
            f"Resume analysis failed: "
            f"{type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Analysis failed: {type(exc).__name__}. "
                "Check the FastAPI terminal."
            )
        ) from exc

    finally:
        await resume.close()


@app.post("/analyze-text")
def analyze_text(request: TextAnalysisRequest):
    try:
        return analyze_resume(
            request.resume_text,
            request.job_description
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        print(
            f"Text analysis failed: "
            f"{type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Analysis failed: {type(exc).__name__}. "
                "Check the FastAPI terminal."
            )
        ) from exc
