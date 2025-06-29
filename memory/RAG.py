import numpy as np
import json
import os
import pickle
from typing import List, Dict, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class RAGDatabase:
    def __init__(self, fields_to_embed: List[str], model_name: str = 'all-MiniLM-L6-v2', storage_dir: str = './rag_db', max_workers: int = 4):

        self.fields = fields_to_embed
        self.model = SentenceTransformer(model_name)
        self.storage_dir = storage_dir
        self.documents_file = os.path.join(storage_dir, 'documents.json')
        self.embeddings_file = os.path.join(storage_dir, 'embeddings.npy')
        self.max_workers = max_workers


        os.makedirs(storage_dir, exist_ok=True)


        self.documents = self._load_documents()
        self.embeddings = self._load_embeddings()


    def _load_documents(self) -> List[Dict]:

        if os.path.exists(self.documents_file):
            with open(self.documents_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _load_embeddings(self) -> List[np.ndarray]:

        if os.path.exists(self.embeddings_file):
            return list(np.load(self.embeddings_file))
        return []

    def _save_documents(self):

        with open(self.documents_file, 'w', encoding='utf-8') as f:
            print('save success in', self.documents_file)
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

    def _save_embeddings(self):

        embeddings_array = np.array(self.embeddings)
        np.save(self.embeddings_file, embeddings_array)
        print('save success in', self.embeddings_file)

    def save(self):
 
        self._save_documents()
        self._save_embeddings()

    def add_document(self, document: Dict):


        if not all(field in document for field in self.fields):
            raise ValueError(f"Document missing required fields: {self.fields}")

        text = " ".join(str(document[field]) for field in self.fields)

        embedding = self.model.encode([text])[0]

        self.documents.append(document)
        self.embeddings.append(embedding)

        self.save()

    def add_documents(self, documents: List[Dict]):

        for doc in documents:

            self.documents.append(doc)
            text = " ".join(str(doc[field]) for field in self.fields)
            embedding = self.model.encode([text])[0]
            self.embeddings.append(embedding)

        self.save()

    def _single_search(self, query: str, k: int = 5) -> List[Dict]:
        if not self.embeddings:
            return []

        query_embedding = self.model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_k_indices:
            results.append({
                'document': self.documents[idx],
                'score': float(similarities[idx])
            })

        return results

    def search(self, query: Union[str, List[str]], k: int = 5,
               batch_size: int = 32, show_progress: bool = True) -> Union[List[Dict], List[List[Dict]]]:

        if isinstance(query, str):
            return self._single_search(query, k)


        queries = query
        total_queries = len(queries)
        results = [None] * total_queries  


        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:

            future_to_idx = {}
            for idx, q in enumerate(queries):
                future = executor.submit(self._single_search, q, k)
                future_to_idx[future] = idx

            if show_progress:
                pbar = tqdm(total=total_queries, desc="Processing queries")

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = []
                    print(f"Error processing query {idx}: {str(e)}")

                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()

        return results

    def clear(self):

        self.documents = []
        self.embeddings = []

        # 删除存储文件
        if os.path.exists(self.documents_file):
            os.remove(self.documents_file)
        if os.path.exists(self.embeddings_file):
            os.remove(self.embeddings_file)

    def __len__(self):
        return len(self.documents)

