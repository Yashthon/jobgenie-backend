import io
import os
# Fix Render proxy issue BEFORE any imports
for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    if var in os.environ:
        print(f"Removing Render proxy var: {var}")
        os.environ.pop(var, None)
import re
import httpx
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
# Load environment variables
# ===============================
load_dotenv()

# Render fix: Remove proxy vars that break OpenAI client
# os.environ.pop("HTTPS_PROXY", None)
# os.environ.pop("HTTP_PROXY", None)

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in environment variables!")

# Use custom httpx client WITHOUT proxy (Render-safe)
custom_http_client = httpx.Client(
    proxies=None,
    timeout=httpx.Timeout(60.0),
    follow_redirects=True
)

client = OpenAI(api_key=OPENAI_API_KEY)

# ===============================
# Initialize FastAPI app
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
# Clean & reliable PDF text extraction
# ===============================
def extract_clean_text_from_pdf(file_bytes: bytes) -> str:
    try:
        raw_text = extract_text(io.BytesIO(file_bytes))
        text = re.sub(r"[^\x20-\x7E\n\r\t]+", " ", raw_text)
        text = re.sub(r"\s+", " ", text).strip()

        if not text or len(text) < 50:
            raise HTTPException(
                status_code=400,
                detail="No readable text found in PDF. The file might be scanned or image-based."
            )

        print(f"Extracted {len(text)} characters of text from resume.")
        return text

    except Exception as e:
        print("PDF extraction error:", e)
        raise HTTPException(status_code=500, detail="Error while extracting text from PDF.")

# ===============================
# Resume parsing (using spaCy)
# ===============================
nlp = spacy.load("en_core_web_sm")

def parse_resume(text: str) -> dict:
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART"]]
    return {
        "skills": list(set(skills))[:10],
        "summary": text[:4000]
    }

# ===============================
# Embedding generator (chunked)
# ===============================
def get_embedding(text: str) -> np.ndarray:
    text = text.strip()
    if not text:
        return np.zeros(1536)

    chunks = textwrap.wrap(text, width=4000)
    embeddings = []

    for i, chunk in enumerate(chunks):
        try:
            print(f"Embedding chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            )
            embeddings.append(np.array(response.data[0].embedding))
        except Exception as e:
            print("Embedding error:", e)
            continue

    if not embeddings:
        raise HTTPException(status_code=500, detail="Failed to generate embeddings from text.")

    return np.mean(embeddings, axis=0)

# ===============================
# Fetch real jobs from Adzuna
# ===============================
def fetch_jobs_adzuna(query: str, location: str):
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
            print("Unexpected Adzuna response:", data)
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

        print(f"Retrieved {len(jobs)} jobs from Adzuna for '{query}' in '{location}'.")
        return jobs

    except Exception as e:
        print("Adzuna fetch error:", e)
        return []

# ===============================
# Upload Endpoint (resume + role + location)
# ===============================
@app.post("/upload_resume/")
async def upload_resume(
    file: UploadFile = File(...),
    role: str = Form(...),
    location: str = Form(...)
):
    contents = await file.read()
    print(f"Received file: {file.filename} ({len(contents)/1024:.2f} KB)")

    text = extract_clean_text_from_pdf(contents)
    parsed = parse_resume(text)
    resume_emb = get_embedding(parsed["summary"])
    jobs = fetch_jobs_adzuna(query=role, location=location)

    if not jobs:
        raise HTTPException(status_code=404, detail="No jobs found for given role and location.")

    for job in jobs:
        job_emb = get_embedding(job["description"])
        job["score"] = float(cosine_similarity([resume_emb], [job_emb])[0][0])

    top_jobs = sorted(jobs, key=lambda x: x["score"], reverse=True)[:10]

    return {
        "extracted_text_length": len(text),
        "skills": parsed["skills"],
        "role": role,
        "location": location,
        "matches": top_jobs
    }

# ===============================
# Health Check
# ===============================
@app.get("/")
def root():
    return {"message": "JobGenie.ai backend (Adzuna + OpenAI) is running successfully!"}
