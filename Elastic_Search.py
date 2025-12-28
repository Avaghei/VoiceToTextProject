from elasticsearch import Elasticsearch
from NormalizeTexts import NormalizeTexts
import math

class Elastic_Search:
    def __init__(self, data, url, username, password,
                 verify_certs=False,
                 index_name='documents',
                 normalizing=True):
        
        self.__es = Elasticsearch(url,basic_auth=(username, password),verify_certs=verify_certs)


        self.__index_name = index_name
        self.__data = data
        self.__normalizing = normalizing


        if self.__normalizing:
            self.__normalizer = NormalizeTexts()

    def combine_headers(self, headers):
        try:
            if not headers:
                return None
            combined = {}
            all_ids = set().union(*(self.__data[h].keys() for h in headers))
            for i in all_ids:
                parts = []
                for h in headers:
                    v = self.__data.get(h, {}).get(i, "")
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        v = ""
                    parts.append(str(v))
                combined[i] = " ".join(parts)
            return combined
        except Exception as e:
            raise e

    def check_index(self):
        try:
            if self.__es.indices.exists(index=self.__index_name):
                return True
            else:
                return False
        except Exception as e:
            raise e
    
    def delete_index(self):
        try:
            if self.__es.indices.exists(index=self.__index_name):
                self.__es.indices.delete(index=self.__index_name)
                print(f"Index '{self.__index_name}' deleted.")
                return True
            else:
                print(f"Index '{self.__index_name}' does not exist.")
                return False
        except Exception as e :
            raise e
    

    def create_index(self, headers, combine_header=True):

        try:
            if combine_header:
                if len(headers) > 1:
                    combined = self.combine_headers(headers)
                    if combined:
                        self.__data['combined_header'] = combined
                        headers = headers + ['combined_header']

            body = {
                "settings": {
                    "analysis": {
                        "char_filter": {
                            "fa_norm": {
                                "type": "mapping",
                                "mappings": [
                                    "ي => ی",
                                    "ك => ک",
                                    "آ => ا",
                                    "أ => ا",
                                    "إ => ا",
                                    "ٱ => ا"
                                ]
                            },
                            "fa_phonetic": {
                                    "type": "mapping",
                                    "mappings": [
                                        "ص => س",
                                        "ث => س",
                                        "ذ => ز",
                                        "ض => ز",
                                        "ظ => ز",
                                        "ق => ک",
                                        "غ => ک",
                                        "گ => ک",
                                        "ح => ه",
                                        "ط => ت",
                                        "آ => ا",
                                        "أ => ا",
                                        "إ => ا",
                                        "ي => ی",
                                        "ك => ک",
                                        "ة => ه",
                                        "پ => ب",
                                        "ژ => ش"
                                    ]
                                },
                            "remove_space": {
                                "type": "pattern_replace",
                                "pattern": "\\s+",
                                "replacement": ""
                            }
                        },
                        "tokenizer": {
                            "ngram_tok": {
                                "type": "ngram",
                                "min_gram": 2,
                                "max_gram": 3
                            },#
                            "nospace_ngram_tok": {
                                "type": "ngram",
                            "min_gram": 2,
                                "max_gram": 3
                                }
                        },#
                        "analyzer": {
                            "fa_text": {
                                "tokenizer": "standard",
                                "char_filter": ["fa_norm", "fa_phonetic"],
                                "filter": ["lowercase"]
                            },
                            "fa_text_nospace": {
                                "tokenizer": "nospace_ngram_tok", #standard
                                "char_filter": ["fa_norm", "fa_phonetic", "remove_space"],
                                "filter": ["lowercase"]
                            },
                            "fa_ngram": {
                                "tokenizer": "ngram_tok",
                                "char_filter": ["fa_norm", "fa_phonetic"],
                                "filter": ["lowercase"]
                            },
                            "fa_text_rescore": {
                            "tokenizer": "standard",
                            "char_filter": ["fa_norm", "fa_phonetic"], 
                            "filter": ["lowercase"]
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {}
                }
            }

            for field in headers:
                body["mappings"]["properties"][field] = {
                    "type": "text",
                    "analyzer": "fa_text",
                    "fields": {
                        "nospace": {
                            "type": "text",
                            "analyzer": "fa_text_nospace"
                        },
                        "ngram": {
                            "type": "text",
                            "analyzer": "fa_ngram"
                        },
                        "keyword": {
                            "type": "keyword"
                        },
                        "rescore": {
                        "type": "text",
                        "analyzer": "fa_text_rescore"  
            },
                    }   
                }

            check = self.check_index()
            if check : 
                print(f"Index '{self.__index_name}' already exists.")
            if not check: 
                self.__es.indices.create(index=self.__index_name, body=body)
                print(f"Index '{self.__index_name}' created.")
            

            all_ids = set()
            for v in self.__data.values():
                if isinstance(v, dict):
                    all_ids.update(v.keys())

            for _id in all_ids:
                doc = {}
                for f in headers:
                    val = self.__data.get(f, {}).get(_id, "")
                    if val is None or (isinstance(val, float) and math.isnan(val)):
                        val = ""
                    doc[f] = str(val)
                self.__es.index(index=self.__index_name, id=_id, document=doc)

            self.__es.indices.refresh(index=self.__index_name)
            print(f"Index '{self.__index_name}' ready.")
            return True
        except Exception as e :
            raise e
  
    def search_name(self, asr_text, field='name'):
        try:
            fuzziness=1
            ws=0
            size=5
            if self.__normalizing:
                asr_text = self.__normalizer.normalize_names(asr_text)
                asr_text = self.__normalizer.phonetic_normalize(asr_text)
            asr_text_nospace = asr_text.replace(' ', '')
            query = {
                    "query": {
                        "bool": {
                            "should": [
                                {"term": {f"{field}.keyword": {"value": asr_text, "boost": 15}}}, #8
                                {"match": {f"{field}.nospace": {"query": asr_text_nospace, "boost": 6, "fuzziness": 0}}},#15
                                {"match": {field: {"query": asr_text,"fuzziness": fuzziness,"boost": 13}}}, # 5
                                {"match": {f"{field}.ngram": {"query": asr_text,"fuzziness": 0 ,"boost": 1}}}#2
                            ],
                            "minimum_should_match": 1
                        }
                    }
                }
            res = self.__es.search(index=self.__index_name, body=query, size=size)


           
           


            if 'score' not in self.__data:
                self.__data['score'] = {
                    i: 0.0 for i in self.__data[field].keys()
                }

            scores = [h['_score'] for h in res['hits']['hits']]
            diff_scores = [
                abs(scores[i+1] - scores[i])
                for i in range(len(scores)-1)
            ]

            maxim_diff = max(diff_scores) if diff_scores else 0
            max_score = max(scores) if scores else 0
            threshold = max_score - maxim_diff - ws

            indices = []
            for h in res['hits']['hits']:
                if h['_score'] >= threshold:
                    i = int(h['_id'])
                    indices.append(i)
                    self.__data['score'][i] = h['_score']

            return ({c: {i: self.__data[c][i] for i in indices} for c in self.__data}, max_score)
        except Exception as e:
            raise e

    def rerank_names(self,asr_text,allowed_ids,field='name'):
        try:
            ws=0.05
            size=50
            fuzziness=1
            if not allowed_ids:
                return None

            if self.__normalizing:
                    asr_text = self.__normalizer.normalize_names(asr_text)
                    asr_text = self.__normalizer.phonetic_normalize(asr_text)

            
            query = {
                        "query": {
                            "bool": {"must": [{"terms": {"_id": allowed_ids}}]}
                        },
                        "rescore": {
                            "window_size": size,
                            "query": {
                                "rescore_query": {
                                    "bool": {
                                        "should": [
                                            {"match": {f"{field}.rescore": {"query": asr_text, "fuzziness": fuzziness, "minimum_should_match": "80%"}}},
                                            {"match": {f"{field}.nospace": {"query": asr_text.replace(' ', ''), "fuzziness": fuzziness, "minimum_should_match": "80%"}}}
                                        ]
                                    }
                                },
                                "query_weight": 0,
                                "rescore_query_weight": 1.0
                            }
                        },
                        "size": size
                    }
            
            
            res = self.__es.search(index=self.__index_name, body=query)
            if 'score' not in self.__data:
                self.__data['score'] = {
                    i: 0.0 for i in self.__data[field].keys()
                }

            scores = [h['_score'] for h in res['hits']['hits']]
            diff_scores = [
                abs(scores[i+1] - scores[i])
                for i in range(len(scores)-1)
            ]

            maxim_diff = max(diff_scores) if diff_scores else 0
            max_score = max(scores) if scores else 0
            threshold = max_score - maxim_diff - ws

            indices = []
            for h in res['hits']['hits']:
                if h['_score'] >= threshold:
                    i = int(h['_id'])
                    indices.append(i)
                    self.__data['score'][i] = h['_score']

            return  {c: {i: self.__data[c][i] for i in indices} for c in self.__data}
        except Exception as e:
            raise e
        
        
