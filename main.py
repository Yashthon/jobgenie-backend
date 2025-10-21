import os

# ✅ Remove proxy environment variables (Render sometimes injects these)
for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    if var in os.environ:
        print(f"⚙️ Removing Render proxy var: {var}")
        os.environ.pop(var, None)

import io
import re
import textwrap
import numpy as np
import requests
import spacy
import httpx
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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("❌ Missing OPENAI_API_KEY in environment variables!")

# ✅ Safe OpenAI client (no proxies arg)
client = OpenAI(api_key=OPENAI_API_KEY)

# ===============================
# 2️⃣ Initialize FastAPI app
# ===============================
app = FastAPI(title="JobGenie.ai", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# 3️⃣ Extract text from PDF
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

        print(f"✅ Extracted {len(text)} characters from resume.")
        return text
    except Exception as e:
        print("⚠️ PDF extraction error:", e)
        raise HTTPException(status_code=500, detail="Error extracting text from PDF.")

# ===============================
# 4️⃣ Resume parsing (spaCy)
# ===============================
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def parse_resume(text: str) -> dict:
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART"]]
    return {
        "skills": list(set(skills))[:10],
        "summary": text[:4000]
    }

# ===============================
# 5️⃣ Embedding generator
# ===============================
def get_embedding(text: str) -> np.ndarray:
    text = text.strip()
    if not text:
        return np.zeros(1536)

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
        raise HTTPException(status_code=500, detail="Failed to generate embeddings.")
    return np.mean(embeddings, axis=0)

# ===============================
# 6️⃣ Fetch jobs from Adzuna
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
        response = requests.get(url, timeout=10)
        data = response.json()

        if "results" not in data:
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
# 7️⃣ Upload resume endpoint
# ===============================
@app.post("/upload_resume/")
async def upload_resume(
    file: UploadFile = File(...),
    role: str = Form(...),
    location: str = Form(...)
):
    contents = await file.read()
    print(f"📄 Received file: {file.filename} ({len(contents)/1024:.2f} KB)")

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
# 8️⃣ Health check
# ===============================
@app.get("/")
def root():
    return {"message": "✅ JobGenie.ai backend is running successfully!"}

# ===============================
# 9️⃣ Render entry point
# ===============================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # Render dynamically injects PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
