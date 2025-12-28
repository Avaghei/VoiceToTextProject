from CalculateSimilarities import CalculateSimilarities
from Elastic_Search import Elastic_Search
       
class Search:
    def __init__(self,data ,url , username , password, verify_certs,index_name, normalizing=True):
        
    
            

            self.__es_client = Elastic_Search(
                                data=data,
                                url=url,
                                username=username,
                                password=password,
                                verify_certs=verify_certs,
                                index_name=index_name,
                                normalizing=normalizing
                            )
    def check_index(self):
        try:
            return self.__es_client.check_index()
        except Exception as e :
               print(f"Can't check the index: {type(e).__name__}: {e}")
               return None

    def delete_index(self):
        try:
            return self.__es_client.delete_index()
        except Exception as e:
            print(f"Can't delete the index: {type(e).__name__}: {e}")
            return None
    def create_index(self,headers,combine_header):
        try : 
            self.__es_client.create_index(headers=headers,combine_header=combine_header)
        except Exception as e :
            print(f"Can't create the index: {type(e).__name__}: {e}")
            return None
    def search(self,q,header='full_information'):
       
        try:  
        
            n, max_score = self.__es_client.search_name(q, header)  
            scores = {i: score / max_score for i, score in n['score'].items()} 

            threshold = 0.90
            filtered_ids = [i for i, s in scores.items() if s >= threshold]

            if not filtered_ids:
                filtered_data = {}
            else:
               
                filtered_data = {col: {i: n[col][i] for i in filtered_ids} for col in n if col != 'score'}
                filtered_data['score'] = {i: n['score'][i] for i in filtered_ids}

               
                final_data = {
                    col: {i: filtered_data[col][i] for i in filtered_ids if filtered_data['score'][i] >= 180}
                    for col in filtered_data
                }

           
            idxs = list(final_data[header].keys())
           

           
            rerank_results = self.__es_client.rerank_names(q, allowed_ids=idxs, field=header)


            final_results = {col : {i : rerank_results[col][i] for i in rerank_results[header].keys() if rerank_results['score'][i]>0} for col in rerank_results}

            return  final_results

        except Exception as e:
            print(f"There is problem in search method: {type(e).__name__}: {e}")
            return None

          
    
                                

