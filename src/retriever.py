import os
from google import genai
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class SimpleRetriever:
    def __init__(self):
        api_key=os.getenv("GEMINI_API_KEY")
        self.client=genai.Client(api_key=api_key)
        self.chunks=[]
        self.vectors=[]

    def load_documents(self, folder_path:str):
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                filepath= os.path.join(folder_path, filename)
                with open (filepath, "r", encoding="utf-8")as f:
                    text=f.read()
                    lines=[line.strip()for line in text.split("\n") if line.strip()]
                    self.chunks.extend(lines)
        print (f"✅ Loaded {len(self.chunks)} chunks from documents.")

    def build_vectors(self):
        if not self.chunks:
            print ("⚠️ No chunks to vectorize.")
            return
        self.vectors=[]
        for chunk in self.chunks:
            response= self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=chunk
            )
            vector=np.array(response.embeddings[0].values)
            self.vectors.append(vector)

        self.vectors=np.array(self.vectors)
        print (f"✅ Built vectors for {len(self.vectors)} chunks.")

    def search(self, query:str,top_k:int=2):
        if not self.chunks or len(self.vectors)==0:
            return[]

        response=self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=query
        )
        query_vector=np.array(response.embeddings[0].values)
        query_norm=np.linalg.norm(query_vector)
        vector_norms=np.linalg.norm(self.vectors, axis=1)
        denom=vector_norms*query_norm
        denom=np.where(denom==0, 1e-10, denom)
        similarities = np.dot(self.vectors, query_vector) / denom

        top_indices=np.argsort(similarities)[::-1][:top_k]

        results=[]
        for idx in top_indices:
            results.append({
                "text": self.chunks[idx],
                "score": float(similarities[idx])
            })
        return results

