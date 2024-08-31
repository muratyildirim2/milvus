from pymilvus import MilvusClient, model
from flask import Flask, request, jsonify
import numpy as np
app = Flask(__name__)

client = MilvusClient("milvus_demo.db")

collection_name = "demo_collection"
if not client.has_collection(collection_name=collection_name):
    client.create_collection(
        collection_name=collection_name,
        dimension=768,  
    )

embedding_fn = model.DefaultEmbeddingFunction()

@app.route('/insert', methods=['POST'])
def insert_data():
    docs = request.json.get('docs', [])
    if not docs:
        return jsonify({"error": "Herhangi bir doküman sağlanmadı"}), 400

    vectors = embedding_fn.encode_documents(docs)
    data = [
        {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
        for i in range(len(vectors))
    ]

    res = client.insert(collection_name=collection_name, data=data)

    if hasattr(res, 'insert_count') and hasattr(res, 'ids'):
        response_data = {
            "insert_count": res.insert_count,
            "ids": list(res.ids),
            "cost": res.cost
        }
    else:
        response_data = {
            "message": "Veriler başarıyla eklendi.",
            "raw_response": str(res)
        }
    
    return jsonify(response_data)

@app.route('/query', methods=['GET'])
def query_data():
    try:
        res = client.query(
            collection_name=collection_name,
            filter="",  
            output_fields=["id", "text", "subject"],  
            limit=100  
        )
        
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/search', methods=['POST'])
def search_data():
    query_text = request.json.get('query', '')
    if not query_text:
        return jsonify({"error": "No query provided"}), 400
    
    query_vector = embedding_fn.encode_queries([query_text])

    res = client.search(
        collection_name=collection_name,
        data=query_vector,
        limit=2,
        output_fields=["text", "subject"],
    )

    return jsonify(res)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
