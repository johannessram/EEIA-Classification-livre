from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union, List
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

from embedder___ import embed  # Reuse your existing embedding function

FILENAME = "knn_model.joblib"
loaded_model = joblib.load(FILENAME)
from MongoDBWrapper import MongoDBWrapper

CLUSTER_NAMES = dict_classification = {
    0: "classic children's literature and general fiction",
    1: "non-fiction, history, and contemporary literature.",
    2: "fantasy, science fiction, and dystopian fiction.",
    3: "classic novels and historical fiction",
    4: "young adult literature and fantasy.",
    5: "modern classic literature.",
    6: "thriller and mystery.",
    7: "non-fiction, psychology, and self-help",
    8: "contemporary romance.",
    9: "science fiction and space opera",
    10: "fables and philosophical novels.",
    11: "children's literature and picture books.",
    12: "comedy and humorous science fiction.",
    13: "epic fantasy."
}

MONGODB = MongoDBWrapper(
    db_name="eeia",
    collection_name="BooksCleanEmbedded",
)

app = FastAPI(title="KNNModel model prediction")


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to a list of allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str

class BookMinimalInfo(BaseModel):
    title: str
    description: str

class Suggestions(BaseModel):
    data: List[BookMinimalInfo]

@app.post("/predict")
def predict(req: PredictRequest):
    # Ensure list format
    if isinstance(req.text, str):
        sentences = [req.text]
    else:
        sentences = req.text

    # Embed
    vectors = embed(sentences)
    # print(type(vectors))
    vectors = vectors.tolist()
    # print(type(vectors))
    # print(len(vectors))
    # print(vectors[0])

    # Predict clusters
    predictions = loaded_model.predict(vectors)
    m = loaded_model.predict_proba(vectors)
    # print(m, np.sum(m))
    score = round(np.max(loaded_model.predict_proba(vectors)), 2)

    # Return in same structure as input
    # print(int(predictions[0]))
    return {
        "cluster": CLUSTER_NAMES.get(int(predictions[0]), "Fairy Tale"),
        "score": score
    }



@app.post("/suggest")
def suggest(req: PredictRequest) -> Suggestions:
    prediction = predict(req)
    print(prediction)
    cluster_number= [key for key, value in CLUSTER_NAMES.items() if value == prediction.get('cluster')]
    print(cluster_number)
    if len(cluster_number) == 1:
        cluster_number = cluster_number[0]
    else:
        cluster_number = -1

    print("cluster_number", cluster_number)
    books = MONGODB.find_all(
        filter_dict={
            "cluster": cluster_number
        },
        limit = 5
    )
    print(books)

    if len(books) == 0:
        return Suggestions(data=[])

    books = [{"title": book.get('title', ''), "description": book.get('description', '')} for book in books]

    print(books)
    return Suggestions(data=books)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
