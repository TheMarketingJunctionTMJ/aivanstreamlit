# Streamlit Blog Studio

A Streamlit-based internal AI blog writing tool with an improved workflow:

1. Set blog topic, audience, tone, keywords, and target word count
2. Add facts and quotes
3. Upload a research file
4. Generate an outline first
5. Generate each section one by one
6. Revise individual sections
7. Export the final blog as DOCX

## Local run

```bash
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

## Required environment variables

- `ANTHROPIC_API_KEY`
- `ANTHROPIC_MODEL` (optional)

## Deploy options

- Streamlit Community Cloud
- Render
- Railway

## Notes

- This app is designed for internal use.
- Uploaded files supported: PDF, DOCX, TXT, CSV, XLSX.
- The AI uses an outline-first, section-by-section writing workflow for better blog quality.
