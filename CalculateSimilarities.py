import ast
import heapq
from rapidfuzz import fuzz , distance
from CreateEmbedding import EmbeddingModel
from NormalizeTexts import NormalizeTexts
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class CalculateSimilarities:
    def __init__(self,normalizing=True , cohere_model ="embed-multilingual-light-v3.0" , API_KEY = None , 
                use_cohere=False , use_hugging_face=False , hugging_face_model_path = None , max_retries = 5,
                request_timeout = 20 ):
        self.__use_cohere = use_cohere
        self.__use_hugging_face = use_hugging_face
        self.__normalizing = normalizing

        if self.__use_cohere or self.__use_hugging_face : 
            self.__embedding_model = EmbeddingModel(API_KEY=API_KEY,cohere_model=cohere_model,use_cohere=use_cohere,hugging_face_model_path = hugging_face_model_path,
                                              max_retries=max_retries , request_timeout=request_timeout ).getModel()
        if self.__normalizing : 
            self.__normalizer = NormalizeTexts()

    def combined_score(self,query ,entity,a1,a2,a3):    
        try:
            score1 = fuzz.token_set_ratio(query, entity)/100
            score2 = fuzz.partial_ratio(query, entity)/100
            score3 = distance.Levenshtein.normalized_similarity(query,entity)
            total_score = a1*score1 + a2*score2 + a3*score3
            return total_score
        except Exception as e:
            raise e 

    def CalculateCombinationScore(self,query,data,header,k=10,a1=0.4,a2=0.4,a3=0.2):
        try:
            if self.__normalizing : 
                    query = self.__normalizer.normalize_names(query)
                    query = self.__normalizer.phonetic_normalize(query)
            if 'score' not in data : 
                    data['score'] = {i: 0.0 for i in data[header].keys()}
            
            indices = []
            for idx , entity in data[header].items():
                    idx = int(idx)   
                    indices.append(idx)

                    score = self.combined_score(query,entity,a1,a2,a3)

                    data['score'][idx]=score

            top_k_indices = heapq.nlargest(k,indices,key=lambda i: data["score"][i])

            top_k_data = {col : {i:data[col][i] for i in top_k_indices } for col in data}

            return (top_k_data , top_k_data['score'][top_k_indices[0]])
                
        except Exception as e:
            raise e  

    def calculateCosineSimilarity(self, query, data, header, k=10):
        try:
            if self.__use_hugging_face or self.__use_cohere: 

                if self.__normalizing: 
                    query = self.__normalizer.normalize_names(query)

                if 'score' not in data: 
                    data['score'] = {i: 0.0 for i in data[header].keys()}

                indices = []
                query_embed = np.array(self.__embedding_model.embed_query(query)).reshape(1, -1)

                for idx, entity in data[header].items():
                    idx = int(idx)

                   
                    if isinstance(entity, str):
                        try:
                            if entity.startswith("[") or entity.startswith("("):
                                entity = ast.literal_eval(entity)
                            else:
                                
                                continue
                        except Exception:
                            continue

                    
                    if not isinstance(entity, (list, np.ndarray)):
                        continue

                    try:
                        doc_embed = np.asarray(entity, dtype=float).reshape(1, -1)
                    except ValueError:
                        
                        continue

                    if doc_embed.shape[1] != query_embed.shape[1]:
                        continue

                    score = cosine_similarity(query_embed, doc_embed)[0][0]
                    data['score'][idx] = score
                    indices.append(idx)

                
                top_k_indices = heapq.nlargest(k, indices, key=lambda i: data["score"][i])

               
                top_k_data = {col: {i: data[col][i] for i in top_k_indices} for col in data}

                return (top_k_data, top_k_data['score'][top_k_indices[0]])
            else:
                print("There is problem in using embedding model")
                return None
        
        except Exception as e : 
            raise e 
        