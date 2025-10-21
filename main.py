import io
import os
import re
import textwrap
import numpy as np
import requests
import spacy
from pdfminer.high_level import extract_text
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# 1️⃣ Load environment variables
# ===============================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===============================
# 2️⃣ Initialize FastAPI app
# ===============================
app = FastAPI(title="JobGenie.ai", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# 3️⃣ Clean & reliable PDF text extraction
# ===============================
def extract_clean_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts text content from a PDF file, ignoring images, vectors, or graphics.
    Automatically handles large PDFs and cleans unnecessary characters.
    """
    try:
        # Extract text using pdfminer (text layer only)
        raw_text = extract_text(io.BytesIO(file_bytes))

        # Clean non-printable and excess whitespace
        text = re.sub(r"[^\x20-\x7E\n\r\t]+", " ", raw_text)
        text = re.sub(r"\s+", " ", text).strip()

        # Handle scanned/image-only resumes
        if not text or len(text) < 50:
            raise HTTPException(
                status_code=400,
                detail="No readable text found in PDF. The file might be scanned or image-based."
            )

        print(f"✅ Extracted {len(text)} characters of text from resume.")
        return text

    except Exception as e:
        print("⚠️ PDF extraction error:", e)
        raise HTTPException(status_code=500, detail="Error while extracting text from PDF.")

# ===============================
# 4️⃣ Resume parsing (using spaCy)
# ===============================
nlp = spacy.load("en_core_web_sm")

def parse_resume(text: str) -> dict:
    doc = nlp(text)
    # Extract potential skill keywords (simple heuristic)
    skills = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART"]]
    return {
        "skills": list(set(skills))[:10],
        "summary": text[:4000]  # allow longer context for embedding
    }

# ===============================
# 5️⃣ Embedding generator (chunked)
# ===============================
def get_embedding(text: str) -> np.ndarray:
    """
    Generates OpenAI embeddings safely for large text inputs by chunking.
    Averages all chunk embeddings for a unified vector.
    """
    text = text.strip()
    if not text:
        return np.zeros(1536)

    # Split text into manageable ~1500-token chunks (~4000 characters)
    chunks = textwrap.wrap(text, width=4000)
    embeddings = []

    for i, chunk in enumerate(chunks):
        try:
            print(f"🔹 Embedding chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            )
            embeddings.append(np.array(response.data[0].embedding))
        except Exception as e:
            print("⚠️ Embedding error:", e)
            continue

    if not embeddings:
        raise HTTPException(status_code=500, detail="Failed to generate embeddings from text.")

    return np.mean(embeddings, axis=0)

# ===============================
# 6️⃣ Fetch real jobs from Adzuna
# ===============================
def fetch_jobs_adzuna(query: str, location: str):
    """
    Fetches live job listings from Adzuna API based on role and location.
    """
    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")

    if not app_id or not app_key:
        raise HTTPException(status_code=500, detail="Adzuna API credentials missing.")

    try:
        url = (
            f"https://api.adzuna.com/v1/api/jobs/in/search/1"
            f"?app_id={app_id}&app_key={app_key}"
            f"&results_per_page=10&what={query}&where={location}"
        )

        response = requests.get(url)
        data = response.json()

        if "results" not in data:
            print("⚠️ Unexpected Adzuna response:", data)
            return []

        jobs = [
            {
                "title": j.get("title", "Unknown Role"),
                "company": j.get("company", {}).get("display_name", "Unknown"),
                "location": j.get("location", {}).get("display_name", "N/A"),
                "description": j.get("description", ""),
                "url": j.get("redirect_url", "#")
            }
            for j in data["results"]
        ]

        print(f"✅ Retrieved {len(jobs)} jobs from Adzuna for '{query}' in '{location}'.")
        return jobs

    except Exception as e:
        print("⚠️ Adzuna fetch error:", e)
        return []

# ===============================
# 7️⃣ Upload Endpoint (resume + role + location)
# ===============================
@app.post("/upload_resume/")
async def upload_resume(
    file: UploadFile = File(...),
    role: str = Form(...),
    location: str = Form(...)
):
    """
    Uploads resume, extracts text, parses key data, and fetches job matches.
    """
    contents = await file.read()

    # No size restriction — handle even large resumes gracefully
    print(f"📄 Received file: {file.filename} ({len(contents)/1024:.2f} KB)")

    # Extract clean text
    text = extract_clean_text_from_pdf(contents)

    # Parse resume details
    parsed = parse_resume(text)

    # Generate resume embedding
    resume_emb = get_embedding(parsed["summary"])

    # Fetch live jobs based on user inputs
    jobs = fetch_jobs_adzuna(query=role, location=location)

    if not jobs:
        raise HTTPException(status_code=404, detail="No jobs found for given role and location.")

    # Compare similarity
    for job in jobs:
        job_emb = get_embedding(job["description"])
        job["score"] = float(cosine_similarity([resume_emb], [job_emb])[0][0])

    # Sort top matches
    top_jobs = sorted(jobs, key=lambda x: x["score"], reverse=True)[:10]

    return {
        "extracted_text_length": len(text),
        "skills": parsed["skills"],
        "role": role,
        "location": location,
        "matches": top_jobs
    }

# ===============================
# 8️⃣ Health Check
# ===============================
@app.get("/")
def root():
    return {"message": "✅ JobGenie.ai backend (Adzuna + OpenAI) is running successfully!"}