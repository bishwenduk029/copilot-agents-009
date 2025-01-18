from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Repo of internal AI agents is ready for use"}
