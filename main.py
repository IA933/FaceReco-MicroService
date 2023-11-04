from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return "Free Palestine!"

@app.post("/register")
def register():
    pass

@app.post("/verify")
def verify():
    pass