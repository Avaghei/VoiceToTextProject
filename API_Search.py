import os
from langchain.prompts import ChatPromptTemplate
from langchain_cohere import ChatCohere
from NormalizeTexts import NormalizeTexts
import json, re
import json
import re
def extract_json_from_text(text):

    match = re.search(r"\[\s*(?:\[[^\]]*\]\s*,?\s*)+\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return []
    return []



class API_Search:
    def __init__(self, cohere_api_key, model_name='command-xlarge-nightly', temperature=0.3,normalizing=True):

        os.environ['COHERE_API_KEY'] = cohere_api_key
        self.__llm = ChatCohere(model=model_name, temperature=temperature)
        self.__normalizing = normalizing
        if self.__normalizing:
            self.__normalizer = NormalizeTexts()
    
    
    def Search(self, names_list, input_name ,k=10):
        if self.__normalizing:
            input_name = self.__normalizer.normalize_names(input_name)
            input_name = self.__normalizer.phonetic_normalize(input_name)
            #print(f"Normalized input name: {input_name}")
            for i in range(len(names_list)):
                names_list[i][0] = self.__normalizer.phonetic_normalize(names_list[i][0])
               # print(
               #     f"Normalized name {i}: {names_list[i][0]}"  
               # )
              
        prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
                    یک اسم ورودی دارای غلط املایی به همراه یک لیست اسامی در اختیارت قرار میگیرد
                   شما اسم ورودی را با اسامی داخل لیست مقایسه کن اگر هیچ ربط تلفظی  نداشتند خروجی 
                   []
                   برگردان در غیر این صورت کل لیست جواب ما می باشد
       """
    ),
    (
        "human",
        """
        اسم ورودی (ممکن است غلط املایی داشته باشد):
        {Input_name}

        لیست اسامی (هر آیتم [نام کامل, آیدی]):
        {Names_List}

        وظیفه:
        ببین ورودی به لیست ربط داره یا نه  و فرمت خروجی به صورت زیر است 

        فرمت خروجی:
        [
        ["<نام کامل دقیق از لیست>", "<آیدی>"],
        ...
        ]
        یا
        []
        """
    )
])

        
    
        messages = prompt.invoke({ "Input_name": input_name, "Names_List": names_list,  }) 
        response = self.__llm.invoke(messages) 
        return extract_json_from_text(response.content)