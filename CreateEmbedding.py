
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import pandas as pd
from langchain_cohere import CohereEmbeddings
import os
from NormalizeTexts import NormalizeTexts
import time

class EmbeddingModel:
    def __init__(self,API_KEY=None,cohere_model=None,use_cohere=False,hugging_face_model_path = None,max_retries=5 , request_timeout=20 ):


        self.__use_cohere=use_cohere
        self.__API_KEY = API_KEY
        self.__hugging_face_model_path = hugging_face_model_path
        self.__max_retires = max_retries 
        self.__request_time_out = request_timeout
        self.__cohere_model = cohere_model

    def getModel(self):
        if self.__use_cohere==True:
            os.environ['COHERE_API_KEY']=self.__API_KEY
            embedding_model = CohereEmbeddings(
            model = self.__cohere_model , 
            max_retries=self.__max_retires , 
            request_timeout=self.__request_time_out
            )
        else : 
            device = "cuda" if torch.cuda.is_available() else "cpu"
            embedding_model = HuggingFaceEmbeddings(
            model_name=self.__hugging_face_model_path, 
            model_kwargs={"device": device}  
            )
        return embedding_model



class CreateEmbedding:
    def __init__(self ,csv_file_path,delimiter=',',cohere_model="embed-multilingual-light-v3.0",encoding = 'utf-8-sig',normalizing=True, API_KEY=None,use_cohere=False,hugging_face_model_path = None,max_retries=5 , request_timeout=20 ):
        self.use_cohere=use_cohere
        self.normalizing = normalizing
        self.df = pd.read_csv(csv_file_path, delimiter=delimiter, encoding=encoding)
        self.length = len(self.df)


        self.embedding_model = EmbeddingModel(API_KEY=API_KEY,cohere_model=cohere_model,use_cohere=use_cohere,hugging_face_model_path = hugging_face_model_path,
                                              max_retries=max_retries , request_timeout=request_timeout).getModel()
        
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
