import pandas as pd
import numpy as np 
from langchain_cohere import CohereEmbeddings
import os
import time 
from hazm import Normalizer
import unicodedata
import re
from hazm import word_tokenize
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from sklearn.metrics.pairwise import cosine_similarity
import ast
import heapq
import Levenshtein



# Verify that the environment variable is set
print(f"FLAIR_CACHE_ROOT environment variable set to: {os.environ.get('FLAIR_CACHE_ROOT')}")
from langchain_cohere import ChatCohere
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from rapidfuzz import fuzz , distance




from flair.models import SequenceTagger
from flair.data import Sentence
import flair
from pathlib import Path

flair.cache_root = Path("D:\models\.flair")# WORKS

print(f"Flair cache root directory: {flair.cache_root}")
class NormalizeTexts : 
    def __init__(self):
        pass
       
    def normalize_names(self,text):
        normalizer = Normalizer()
    
        text = text.lstrip('\ufeff')

        text = normalizer.normalize(text)
        
        
        text = text.replace('\u200c', ' ')  
        
        
        text = unicodedata.normalize('NFC', text)
        
       
        text = re.sub(r'[0-9۰-۹]', '', text)
        
       
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_names(self,text):
        return word_tokenize(text)
    
class NameEntityRecognitionModel:

    def __init__(self, model_name="PooryaPiroozfar/Flair-Persian-NER"):
        # Load the model
        self.tagger = SequenceTagger.load(model_name)

    def getEntities(self, text):
        sentence = Sentence(text)
        self.tagger.predict(sentence)

        # --- Collect entities ---
        entities = {}
        entity_texts = []

        for span in sentence.get_spans("ner"):
            ent_type = span.get_label("ner").value
            if ent_type not in entities:
                entities[ent_type] = []
            entities[ent_type].append(span.text)
            entity_texts.append(re.escape(span.text))  # escape for regex
            print(f"Entity: {span.text}, Type: {ent_type}, Confidence: {span.score}")

        # --- Find unlabeled text by splitting around entities ---
        if entity_texts:
            # Create regex pattern that matches all entities
            pattern = "|".join(entity_texts)
            # Split text at entities; remaining pieces are unlabeled
            unlabelled = [s.strip() for s in re.split(pattern, text) if s.strip()]
        else:
            unlabelled = [text.strip()]

        return {
            "entities": entities,
            "unlabelled": unlabelled
        }

class EmbeddingModel:
    def __init__(self,API_KEY=None,cohere_model=None,use_cohere=False,hugging_face_model_path = None,max_retries=5 , request_timeout=20 ):


        self.use_cohere=use_cohere
        self.API_KEY = API_KEY
        self.hugging_face_model_path = hugging_face_model_path
        self.max_retires = max_retries 
        self.request_time_out = request_timeout
        self.cohere_model = cohere_model

    def getModel(self):
        if self.use_cohere==True:
            os.environ['COHERE_API_KEY']=self.API_KEY
            embedding_model = CohereEmbeddings(
            model = self.cohere_model , 
            max_retries=self.max_retires , 
            request_timeout=self.request_time_out
            )
        else : 
            device = "cuda" if torch.cuda.is_available() else "cpu"
            embedding_model = HuggingFaceEmbeddings(
            model_name=self.hugging_face_model_path,  # local path
            model_kwargs={"device": device}  # send to GPU if available
            )
        return embedding_model
        
class CreateEmbedding:
    def __init__(self ,csv_file_path,delimiter=',',cohere_model="embed-multilingual-light-v3.0",encoding = 'utf-8-sig',normalizing=True, API_KEY=None,use_cohere=False,hugging_face_model_path = None,max_retries=5 , request_timeout=20 ):
        self.use_cohere=use_cohere
        self.normalizing = normalizing
        self.df = pd.read_csv(csv_file_path, delimiter=delimiter, encoding=encoding)
        self.length = len(self.df)


        self.embedding_model = EmbeddingModel(API_KEY=API_KEY,cohere_model=cohere_model,use_cohere=use_cohere,hugging_face_model_path = hugging_face_model_path,
                                              max_retries=max_retries , request_timeout=request_timeout ).getModel()
        
        if self.normalizing :
            self.normalizer = NormalizeTexts()
        

    def CreateEmbeddingFileForHeaders(self, batch_size=150, wait_time=65,
                         verbose=False,
                        header_name_list_for_embedding=[],
                        new_file_name='table_with_embeddings.csv'):

            if verbose:
                print(self.df.head().to_string())
                print(f'Total rows: {self.length}')
                print('The embedding process has started')

            # Initialize id column
            self.df['id'] = range(1, self.length + 1)

            for header_name_for_embedding in header_name_list_for_embedding:

                # Step 0: Normalize the whole column once and optionally overwrite
                if self.normalizing:
                    self.df[header_name_for_embedding] = self.df[header_name_for_embedding].apply(
                        lambda x: self.normalizer.normalize_names(x) if pd.notna(x) else ""
                    )

                # Step 1: Get all unique non-null values
                all_names = self.df[header_name_for_embedding].dropna().unique()
                all_names = [str(name) for name in all_names if str(name).strip() != ""]
                print(f"Embedding {len(all_names)} unique values from column '{header_name_for_embedding}'...")

                name_to_embedding = {}

                # Step 2: Batch-embed unique values
                for i in range(0, len(all_names), batch_size):
                    batch = all_names[i:i + batch_size]
                    batch = [str(x) for x in batch] 
                    batch_embeddings = self.embedding_model.embed_documents(batch)

                    # Store embeddings
                    for name, emb in zip(batch, batch_embeddings):
                        name_to_embedding[name] = emb

                    if verbose:
                        print(f"Embedded batch {i}–{i + len(batch)}")

                    if self.use_cohere and i + batch_size < len(all_names):
                        if verbose:
                            print(f"Waiting {wait_time} seconds before next batch...")
                        time.sleep(wait_time)

                # Step 3: Assign embeddings back to DataFrame
                emb_col_name = f"embeddings_{header_name_for_embedding}"
                print(f"Assigning embeddings to column '{emb_col_name}'...")
                self.df[emb_col_name] = self.df[header_name_for_embedding].map(name_to_embedding)

            # Step 4: Save the DataFrame
            self.df.to_csv(new_file_name, index=False, encoding='utf-8-sig')
            print("Embedding generation completed and saved successfully!")
   
class CalculateSimilarities:
    def __init__(self,csv_file_path,delimiter=',',encoding = 'utf-8-sig',Normalizing=True,cohere_model = "embed-multilingual-light-v3.0",
                 API_KEY=None,use_cohere=False,use_hugging_face=False,hugging_face_model_path = None,max_retries=5 , request_timeout=20):
        self.use_cohere=use_cohere
        self.use_hugging_face=use_hugging_face
        self.df = pd.read_csv(csv_file_path,delimiter=delimiter,encoding=encoding)
        self.normalizing = Normalizing


        self.embedding_model = EmbeddingModel(API_KEY=API_KEY,cohere_model=cohere_model,use_cohere=use_cohere,hugging_face_model_path = hugging_face_model_path,
                                              max_retries=max_retries , request_timeout=request_timeout ).getModel()
        if self.normalizing :
            self.normalizer = NormalizeTexts()
       
        
        
    def calculateCombinationScore(self,query,header,k=50):
        if self.normalizing :
            query  = self.normalizer.normalize_names(query)

        idx_to_similarity = {}

        for idx in self.df.index : 
            string = self.df.at[idx,header]

            score1 = fuzz.token_set_ratio(query, string)
            score2 = fuzz.partial_ratio(query, string)
            score3 = distance.Levenshtein.normalized_similarity(query,string)
            score3 = score3*100
            final_score = 0.2*score1+0.4*score2+0.4*score3 

            idx_to_similarity[idx]=final_score

        top_k = heapq.nlargest(k,idx_to_similarity,key=idx_to_similarity.get)

        top_k_with_scores = [(i,idx_to_similarity[i]) for i in top_k]

        return idx_to_similarity,top_k,top_k_with_scores

    def calculateLevenshteinDistance(self,query,header,k=50):
        if self.normalizing :
            query  = self.normalizer.normalize_names(query)
        

        idx_to_similarity = {}
        for idx in self.df.index : 
            string = self.df.at[idx,header]
            sim = Levenshtein.ratio(query, string)
            idx_to_similarity[idx]=sim

        top_k = heapq.nlargest(k,idx_to_similarity,key=idx_to_similarity.get)

        top_k_with_scores = [(i,idx_to_similarity[i]) for i in top_k]

        return idx_to_similarity,top_k,top_k_with_scores
    
    def calculateCosineSimilarity(self, query, header, k=50):
        if not self.use_hugging_face and not self.use_cohere:
            print("Can't use this method because the embedding model is not specified.")
            return

        if self.normalizing:
            query = self.normalizer.normalize_names(query)

       
        query_embed = np.array(self.embedding_model.embed_query(query)) 
        if query_embed.ndim == 1:
            query_embed = query_embed.reshape(1, -1)

        idx_to_similarity = {}

        for idx in self.df.index:
            value = self.df.at[idx, header]
            if isinstance(value, str):
                value = ast.literal_eval(value)
            doc_embed = np.array(value)
            if doc_embed.ndim == 1:
                doc_embed = doc_embed.reshape(1, -1)
            similarity = cosine_similarity(query_embed, doc_embed)[0][0]
            idx_to_similarity[idx] = similarity

        top_k = heapq.nlargest(k,idx_to_similarity,key=idx_to_similarity.get)

        top_k_with_scores = [(i,idx_to_similarity[i]) for i in top_k]

        return idx_to_similarity,top_k,top_k_with_scores
   
    

        



