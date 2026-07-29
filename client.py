import requests
 
url = "http://127.0.0.1:8000/ask"
 
response = requests.post(
 
    url,
 
    json={
 
        "question":"What is FAISS?"
 
    }
 
)
 
print(response.json())
 