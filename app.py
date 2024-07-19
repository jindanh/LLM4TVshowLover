# data from: https://www.kaggle.com/datasets/oleksiimartusiuk/1500-tv-shows-ranked/data

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()   # load the .env file

loader = CSVLoader(file_path="TV_show_data_summary_only.csv")
documents = loader.load()
print(documents[0])

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)   # vectorize data


'''
    similarity search from the knowledge base
'''
def retrieve_info(query):
    similar_records = db.similarity_search(query, k=3)
    tv_show_names = [record.page_content for record in similar_records]
    return tv_show_names


### example: recommend TV shows based on user query 
user_query = "I like shows with a lot of action and drama."

results = retrieve_info(user_query)
print(results)



# '''
#     set up llm and prompts
# '''
# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# template = """
#     I am looking for a TV show with a lot of action and drama{user_preference}. 
#     Can you recommend a show for me {best_practice}?
# """

# prompt = PromptTemplate(
#     input_variables=["user_preference", "best_practice"],
#     template=template
# )

