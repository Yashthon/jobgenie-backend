import os

# ✅ Disable Render's proxy environment variables before OpenAI client initialization
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
from pdfminer.high_level import extract_text
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

# ===============================
# 1️⃣ Load environment variables
# ===============================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("❌ Missing OPENAI_API_KEY environment variable!")

# ✅ Initialize OpenAI client safely (no proxy issue)
client = OpenAI(api_key=OPENAI_API_KEY)

# ===============================
# 2️⃣ Initialize FastAPI app
# ===============================
app = FastAPI(title="JobGenie.ai", version="4.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "https://jobgenie-frontend.onrender.com",  # your deployed frontend (if Render)
    ],  # allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# 3️⃣ Extract clean text from PDF
# ===============================
def extract_clean_text_from_pdf(file_bytes: bytes) -> str:
    try:
        raw_text = extract_text(io.BytesIO(file_bytes))
        text = re.sub(r"[^\x20-\x7E\n\r\t]+", " ", raw_text)
        text = re.sub(r"\s+", " ", text).strip()

        if not text or len(text) < 50:
            raise HTTPException(
                status_code=400,
                detail="No readable text found. The file may be image-based."
            )
        return text
    except Exception as e:
        print("⚠️ PDF extraction error:", e)
        raise HTTPException(status_code=500, detail="Error reading PDF.")

# ===============================
# 4️⃣ Resume parsing (spaCy)
# ===============================
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def parse_resume(text: str):
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART"]]
    return {"skills": list(set(skills))[:10], "summary": text[:4000]}

# ===============================
# 5️⃣ Embedding generator
# ===============================
def get_embedding(text: str) -> np.ndarray:
    if not text.strip():
        return np.zeros(1536)
    chunks = textwrap.wrap(text, width=4000)
    vectors = []
    for chunk in chunks:
        try:
            response = client.embeddings.create(model="text-embedding-3-small", input=chunk)
            vectors.append(np.array(response.data[0].embedding))
        except Exception as e:
            print("⚠️ Embedding error:", e)
    if not vectors:
        raise HTTPException(status_code=500, detail="Failed to create embeddings.")
    return np.mean(vectors, axis=0)

# ===============================
# 6️⃣ Fetch jobs from Adzuna
# ===============================
def fetch_jobs_adzuna(query: str, location: str):
    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        raise HTTPException(status_code=500, detail="Adzuna API keys missing.")

    try:
        url = f"https://api.adzuna.com/v1/api/jobs/in/search/1?app_id={app_id}&app_key={app_key}&results_per_page=10&what={query}&where={location}"
        res = requests.get(url, timeout=10)
        data = res.json().get("results", [])
        return [
            {
                "title": j.get("title", ""),
                "company": j.get("company", {}).get("display_name", ""),
                "location": j.get("location", {}).get("display_name", ""),
                "description": j.get("description", ""),
                "url": j.get("redirect_url", "#"),
            }
            for j in data
        ]
    except Exception as e:
        print("⚠️ Adzuna API error:", e)
        return []

# ===============================
# 7️⃣ Upload Resume Endpoint
# ===============================
@app.post("/upload_resume/")
async def upload_resume(file: UploadFile = File(...), role: str = Form(...), location: str = Form(...)):
    contents = await file.read()
    text = extract_clean_text_from_pdf(contents)
    parsed = parse_resume(text)
    resume_emb = get_embedding(parsed["summary"])
    jobs = fetch_jobs_adzuna(role, location)

    for job in jobs:
        job_emb = get_embedding(job["description"])
        job["score"] = float(cosine_similarity([resume_emb], [job_emb])[0][0])

    top_jobs = sorted(jobs, key=lambda j: j["score"], reverse=True)[:10]
    return {"skills": parsed["skills"], "matches": top_jobs}

# ===============================
# 8️⃣ Health Check
# ===============================
@app.get("/")
def root():
    return {"status": "✅ JobGenie backend is running!"}

# ===============================
# 9️⃣ Render entry point
# ===============================
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 10000))  # Render gives PORT automatically
#     uvicorn.run(app, host="0.0.0.0", port=port)
