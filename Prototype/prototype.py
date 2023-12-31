import numpy as np
import pandas as pd

import pickle
import os
import json
import PyPDF2 as pdf
from PyPDF2 import PdfReader
from progress.bar import Bar
import re
from tqdm import tqdm
import openai
import time
import warnings
import sys
warnings.filterwarnings('ignore')
from openai.embeddings_utils import get_embedding
from IPython.display import clear_output

os.system('cls' if os.name == 'nt' else 'clear')
openai.api_key = input("Enter API key:")
os.system('cls' if os.name == 'nt' else 'clear')
application_path = os.path.dirname(sys.executable)

def embed(x):
    '''embed with openai'''
    return get_embedding(x, engine="text-embedding-ada-002")

def reset_data():
    '''clear all data from directories and json history'''
    data_json = {"stored_pdfs":{}
    }
    json_object = json.dumps(data_json, indent=4)
    with open(f"{application_path}/data_json.json", "w") as outfile:
        outfile.write(json_object)  
    manual_vectors = pd.DataFrame({"pdf_id":[],"pdf_name":[],"avg_embedding":[]})
    manual_vectors.to_pickle(f'{application_path}/vectors/manual_vectors.csv')
    folder = f"{application_path}/vectors/PageVDs"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            return False
    return True

def load_pdfs(reset=False):
    '''Loads pdfs that have not been loaded into the dataframes. Set reset to True to wipe all data and start again'''
    tqdm.pandas()
    if reset == True:
        reset_data()
    with open(f"{application_path}/data_json.json", 'r') as openfile:
        data_json = json.load(openfile)
    pdf_files = os.listdir(f"{application_path}/pdf_files")
    max_id = len(data_json.keys())
    manual_vectors = pd.read_pickle(f'{application_path}/vectors/manual_vectors.csv')
    count = 0
    for file in pdf_files:
        if file == ".DS_Store":
            continue
        
        if file not in data_json["stored_pdfs"].values():
            count+=1
            cur_id = len(data_json["stored_pdfs"].keys())
            data_json["stored_pdfs"][cur_id] = file
            # PYPDF Part
            cur_pdf_text = []
            print(f"processing {file}")
            with open(f"{application_path}/pdf_files/" + file,"rb") as pdf_file:
                reader = PdfReader(pdf_file)
                for j in range(0, len(reader.pages)):
                    cur_string = reader.pages[j].extract_text()
                    cur_pdf_text.append("|start of page {} ".format(j+1)+cur_string + " end of page {}|".format(j+1))
            page_df = pd.DataFrame({"pg_no":list(range(1,len(cur_pdf_text)+1)),"pg_text":cur_pdf_text,"pdf_id":np.full(len(cur_pdf_text),cur_id)})
            print("embedding vectors...")
            max_count = len(page_df["pg_text"])
            def embed_load(x):
                return embed(x)
            page_df["pg_embedding"] = page_df["pg_text"].progress_apply(embed_load)
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"{file} complete!")
            manual_vectors = manual_vectors.append({"pdf_id":cur_id,"pdf_name":file,"avg_embedding":np.array(list(page_df["pg_embedding"])).mean(axis=0)},ignore_index=True)
            page_df.to_pickle(f'{application_path}/vectors/PageVDs/{cur_id}.csv')
            
        else:
            print(f"{file} is a duplicate file")
    os.system('cls' if os.name == 'nt' else 'clear')
    manual_vectors.to_pickle(f'{application_path}/vectors/manual_vectors.csv')
    json_object = json.dumps(data_json, indent=4)
    with open(f"{application_path}/data_json.json", "w") as outfile:
        outfile.write(json_object)
        
    if count == 0:
        print("No new files detected.")
    elif count == 1:
        print(f"Added {count} new file to the database.")
    else:
        print(f"Added {count} new files to the database.")
    
def get_suitable_documents(prompt, k_docs=8, details=False):
    '''Queries into the vector database to find the most suitable documents based on the prompt'''
    embedded_prompt = embed(prompt)
    manual_df = pd.read_pickle(f'{application_path}/vectors/manual_vectors.csv')
    manual_df["cos"] = manual_df["avg_embedding"].apply(lambda x: x@embedded_prompt)
    manual_df = manual_df.sort_values(by="cos",ascending=False)
    if details == True:
        print("looking at:",list(manual_df["pdf_name"])[0:k_docs])
    return list(manual_df["pdf_id"][0:k_docs])
    
    
def get_page_text(pdf_id,pg_no):
    '''Queries into the database to get the text of a pdf given a page'''
    try:
        page_df = pd.read_pickle(f"{application_path}/vectors/PageVDs/{int(pdf_id)}.csv")
        return page_df[page_df["pg_no"] == pg_no].iloc[0]["pg_text"]
    except:
        return ""
    

    
def get_suitable_text(prompt,k_docs=8, k_pages=10, details=False, threshold = 0.7):
    '''Queries into the vector database to find the most suitable context text based on the prompt'''
    with open(f"{application_path}/data_json.json", 'r') as openfile:
        data_json = json.load(openfile)
    #Find suitable manuals
    embedded_prompt = embed(prompt)
    doc_ids = get_suitable_documents(prompt,k_docs,details)
    relevant_df = pd.DataFrame()
    for doc_id in doc_ids:
        relevant_df = relevant_df.append(pd.read_pickle(f"{application_path}/vectors/PageVDs/{int(doc_id)}.csv"),ignore_index=True)
    #Merge Documents
    relevant_df["pg_cos"] = relevant_df["pg_embedding"].apply(lambda x: np.array(x)@embedded_prompt)
    ranked_filtered_df = relevant_df.sort_values(by="pg_cos",ascending=False)
    idx_pg_pairs = sorted(list(zip(ranked_filtered_df['pdf_id'][0:k_pages],ranked_filtered_df['pg_no'][0:k_pages])))
    idx_pg_dict = {}
    
    for key, value in idx_pg_pairs:
        idx_pg_dict.setdefault(key, []).append(value)
        
    for key,value in idx_pg_dict.items():
        page_list = value
        new_list = []
        for i,page in enumerate(page_list):
            if i == len(page_list)-1:
                new_list.append(page)
                new_list.append(page+1)
                break
            if (page_list[i+1] - page) >= 4:
                new_list.append(page-1)
                new_list.append(page)
                new_list.append(page+1)
            else:
                new_list.append(page-1)
                for pgno in list(range(page,page_list[i+1])):
                    new_list.append(pgno)
        idx_pg_dict[key] = list(set([i for i in new_list if i != 0]))
    if details == True:
        for ID,pages in idx_pg_dict.items():
            print("pages for {}:".format(data_json["stored_pdfs"][str(ID)]),pages)
    final_list = []
    for ID, pages in idx_pg_dict.items():
        for page in pages:
            final_list.append(get_page_text(ID,page))
    return " ".join(final_list)

def chatbot(premise = "You’re a technical support bot that has access to manuals. The manual information is here: ",
            prompt_mod = " Notes: You CAN ONLY USE INFO FROM THE MANUAL TEXT I REPEAT.  \
            You CAN ONLY USE INFO FROM THE MANUAL TEXT. If there is nothing available in \
            the text, SAY I don't have the info. Do not generalize answers. Do not state \
            the page numbers. Remember the answers can be from different manuals. Say \
            'According to my resources' if you understand this along with your response",
            
            k_docs = 8,
            k_pages = 10,
            threshold = 0.7,
            details = False):
    '''Start of the chatbot. Able to modify the premise or prompt'''
    guide = 'Key in "new" to start a new context session \nKey in "exit" to end the chatbot \nKey in "manuals" to display all available manual \nKey in "load pdfs" to load new pdfs or "load pdfs reset" to reset the database (costly) and load'
    messages=[
    {"role": "system", "content": premise}
      ]
    response_list = []
    
    print(guide)
    def conversation(response):
        '''The openai api responses'''
        if len(messages) == 1:
            messages[0]["content"] += get_suitable_text(response,k_pages=k_pages,details=details)
        response_list.append(response)
        content = "Question: " + response + prompt_mod
        messages.append({"role": "user", "content": content})
        
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=messages,
        )
        message = completion["choices"][0]["message"]
        chat_response = completion.choices[0].message.content
        token_usage = completion.usage.total_tokens
        messages.append({"role": "assistant", "content": chat_response})

        return (messages,token_usage)
    
   


    while True:
        cur_msg = input("Question: \n")
        if cur_msg == "exit":
            os.system('cls' if os.name == 'nt' else 'clear')
            break
        elif cur_msg == "manuals":
            os.system('cls' if os.name == 'nt' else 'clear')
            with open(f"{application_path}/data_json.json", 'r') as openfile:
                data_json = json.load(openfile)
            for stored_pdf in list(data_json["stored_pdfs"].values()):
                print(stored_pdf)
            print("")
            input("Press Enter to continue...")
            os.system('cls' if os.name == 'nt' else 'clear')
            print(guide)

            
            
        elif cur_msg == "load pdfs":
            load_pdfs()
            print(guide)
        elif cur_msg == "load pdfs reset":
            load_pdfs(reset=True)
        elif cur_msg == "new":
            os.system('cls' if os.name == 'nt' else 'clear')
            print(guide)
            messages=[{"role": "system", "content": premise}]
        else:
            cur_conv = conversation(cur_msg)
            print("\nChatGPT:\n" + cur_conv[0][-1]["content"])
            print("\n")
            print(f"tokens used: {cur_conv[1]}/16385\n")
            continue
    print("Chat Ended")
        
            
def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    
    chatbot(details=True,k_pages=10,k_docs=12)

main()