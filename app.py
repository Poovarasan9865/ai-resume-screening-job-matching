import json

import requests
import streamlit as st


# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------
st.set_page_config(
    page_title="AI Resume Screening & Job Matching",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI Resume Screening & Job Matching")
st.caption(
    "LLM analysis • Semantic similarity • Skill gap analysis"
)


# ---------------------------------
# SIDEBAR
# ---------------------------------
with st.sidebar:
    st.header("Backend Configuration")

    api_url = st.text_input(
        "FastAPI URL",
        value="http://127.0.0.1:8002"
    ).strip().rstrip("/")

    st.info(
        "Start FastAPI and Ollama in separate terminals "
        "before analyzing resumes."
    )

    if st.button("Check Backend Connection", use_container_width=True):
        try:
            health_response = requests.get(
                f"{api_url}/health",
                timeout=15
            )
            health_response.raise_for_status()

            health_data = health_response.json()

            if health_data.get("status") == "ok":
                st.success("FastAPI is connected.")
            else:
                st.success(
                    f"FastAPI responded: {health_data}"
                )

        except requests.exceptions.Timeout:
            st.error(
                "Health check timed out. Verify the FastAPI "
                "server and configured port."
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to FastAPI. Start the backend "
                "and verify the port."
            )

        except requests.exceptions.RequestException as exc:
            st.error(f"Backend error: {exc}")


# ---------------------------------
# INPUTS
# ---------------------------------
left, right = st.columns(2)

with left:
    st.subheader("1. Upload Resume")

    resume_file = st.file_uploader(
        "Choose a text-based PDF resume",
        type=["pdf"]
    )

    if resume_file is not None:
        st.success(f"Selected: {resume_file.name}")
        st.caption(
            f"File size: {len(resume_file.getvalue()) / 1024:.1f} KB"
        )

with right:
    st.subheader("2. Job Description")

    job_description = st.text_area(
        "Paste the job responsibilities and required skills",
        height=260,
        placeholder=(
            "Example: Python, SQL, FastAPI, Machine Learning, "
            "NLP, LLM, REST APIs..."
        )
    )


# ---------------------------------
# ANALYSIS
# ---------------------------------
if st.button(
    "Analyze Resume",
    type="primary",
    use_container_width=True
):
    if not api_url.startswith(("http://", "https://")):
        st.error("Enter a valid FastAPI URL.")

    elif resume_file is None:
        st.error("Please upload a PDF resume.")

    elif not job_description.strip():
        st.error("Please enter a job description.")

    elif len(resume_file.getvalue()) > 10 * 1024 * 1024:
        st.error("PDF must be 10 MB or smaller.")

    else:
        try:
            with st.spinner(
                "Analyzing resume with embeddings and Ollama..."
            ):
                response = requests.post(
                    f"{api_url}/analyze",
                    files={
                        "resume": (
                            resume_file.name,
                            resume_file.getvalue(),
                            "application/pdf"
                        )
                    },
                    data={
                        "job_description": job_description
                    },
                    timeout=(10, 600)
                )

            if response.ok:
                result = response.json()

                st.success("Resume analysis completed!")

                # Match overview
                st.divider()
                st.subheader("📊 Match Overview")

                col1, col2 = st.columns(2)

                semantic_score = result.get("semantic_match_score")
                skill_coverage = result.get("skill_coverage_score")

                with col1:
                    if semantic_score is not None:
                        score = max(
                            0.0,
                            min(100.0, float(semantic_score))
                        )
                        st.metric(
                            "Semantic Match Score",
                            f"{score:.1f}%"
                        )
                        st.progress(score / 100)
                    else:
                        st.metric("Semantic Match Score", "N/A")

                with col2:
                    if skill_coverage is not None:
                        st.metric(
                            "Detected Skill Coverage",
                            f"{skill_coverage}%"
                        )
                    else:
                        st.metric(
                            "Detected Skill Coverage",
                            "N/A"
                        )

                st.caption(
                    result.get(
                        "score_explanation",
                        "Similarity is not a hiring probability."
                    )
                )

                # Skills
                st.divider()
                st.subheader("🛠️ Skill Gap Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**✅ Matched Skills**")
                    matched_skills = result.get("matched_skills", [])

                    if matched_skills:
                        for skill in matched_skills:
                            st.write(f"✓ {skill}")
                    else:
                        st.info("No matched skills returned.")

                with col2:
                    st.markdown("**📌 Missing Skills**")
                    missing_skills = result.get("missing_skills", [])

                    if missing_skills:
                        for skill in missing_skills:
                            st.write(f"• {skill}")
                    else:
                        st.info("No missing skills returned.")

                # Additional analysis
                st.divider()
                st.subheader("🧠 AI-Generated Analysis")

                st.markdown("**Strengths**")
                strengths = result.get("strengths", [])

                if strengths:
                    for item in strengths:
                        st.write(f"✓ {item}")
                else:
                    st.write("No strengths returned.")

                st.markdown("**Skill Gaps**")
                skill_gaps = result.get("skill_gaps", [])

                if skill_gaps:
                    for item in skill_gaps:
                        st.write(f"• {item}")
                else:
                    st.write("No additional skill gaps returned.")

                st.markdown("**Experience Summary**")
                st.write(
                    result.get("experience_summary")
                    or "No experience summary returned."
                )

                st.markdown("**Recommendation for Human Review**")
                st.write(
                    result.get("recommendation")
                    or "No recommendation returned."
                )

                st.markdown("**Detailed LLM Feedback**")
                st.markdown(
                    result.get("llm_feedback")
                    or "No LLM feedback returned."
                )

                # Interview questions
                questions = result.get("interview_questions", [])

                if questions:
                    with st.expander(
                        "🎯 Interview Preparation Questions"
                    ):
                        for number, question in enumerate(
                            questions, start=1
                        ):
                            st.write(f"{number}. {question}")

                # Model details
                models = result.get("models", {})

                st.divider()
                st.caption(
                    f"Embedding model: "
                    f"{models.get('embedding_model', 'Unknown')} "
                    f"| LLM: {models.get('llm_model', 'Unknown')}"
                )

                st.caption(
                    result.get(
                        "disclaimer",
                        "Human review is required."
                    )
                )

                # Download
                st.download_button(
                    "⬇️ Download Analysis as JSON",
                    data=json.dumps(
                        result,
                        indent=2,
                        ensure_ascii=False
                    ),
                    file_name="resume_analysis.json",
                    mime="application/json",
                    use_container_width=True
                )

            else:
                try:
                    error_data = response.json()
                    detail = error_data.get(
                        "detail",
                        response.text
                    )
                except ValueError:
                    detail = response.text

                st.error(
                    f"API Error ({response.status_code}): {detail}"
                )

                if response.status_code == 503:
                    st.warning(
                        "The backend reports that a required service "
                        "is unavailable. Check Ollama and the configured "
                        "llama3.2 model."
                    )

        except requests.exceptions.ConnectTimeout:
            st.error(
                "Could not establish a connection to FastAPI. "
                "Check the server address and port."
            )

        except requests.exceptions.ReadTimeout:
            st.error(
                "Analysis exceeded the 600-second read timeout. "
                "Check the FastAPI terminal, embedding model, and Ollama."
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to FastAPI. Start the backend "
                "on the configured port and check /health."
            )

        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")

        except (ValueError, TypeError) as exc:
            st.error(f"Invalid response data: {exc}")

        except Exception as exc:
            st.error(
                f"Unexpected error: {type(exc).__name__}: {exc}"
            )
