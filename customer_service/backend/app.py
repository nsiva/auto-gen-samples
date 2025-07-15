from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from logic import lookup_item, check_order_status, process_refund
from dotenv import load_dotenv


load_dotenv()

app = FastAPI()


# CORS for frontend on localhost:4200
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LookupRequest(BaseModel):
    item_name: str

class StatusRequest(BaseModel):
    order_id: str

class RefundRequest(BaseModel):
    order_id: str
    reason: str

@app.post("/lookup")
def lookup(req: LookupRequest):
    result = lookup_item(req.item_name)
    return {"message": result}

@app.post("/status")
def status(req: StatusRequest):
    result = check_order_status(req.order_id)
    return {"message": result}

@app.post("/refund")
def refund(req: RefundRequest):
    result = process_refund(req.order_id, req.reason)
    return {"message": result}

