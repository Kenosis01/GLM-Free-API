# GLM-Free-API (local OpenAI-compatible wrapper)

**It is a reverse engineered API for Deepinfra models. Use it at your own risk and prevent abuse**

If you find this useful, please consider giving a star on GitHub!

This small FastAPI app exposes two endpoints to mimic OpenAI's chat.completions API:

- GET /v1/models — list available models
- POST /v1/chat/completion — accept OpenAI-style requests and return OpenAI-style responses

Requirements:
- Python 3.8+
- Install from requirements.txt

Run locally:

1.Clone this repo:

```powershell
git clone https://github.com/kenosis01/GLM-Free-API.git
```
2.Install dependencies:

```powershell
pip install -r requirements.txt
```
3.Run the app:

```powershell
uvicorn main:app --reload
```
4.Open http://127.0.0.1:8000/docs in your browser to see the Swagger UI.
