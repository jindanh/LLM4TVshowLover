# data from: https://www.kaggle.com/datasets/oleksiimartusiuk/1500-tv-shows-ranked/data

from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import streamlit as st
import json
from datetime import datetime
from typing import List, Dict, Any
import re
import os

load_dotenv()   # load the .env file

class TVShowRAG:
    def __init__(self, openai_api_key: str):
        # Load and process the TV show data
        loader = CSVLoader(file_path="TV_show_data_summary_only.csv")
        documents = loader.load()
        
        # Split the documents into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.db = FAISS.from_documents(docs, embeddings)
    
    def get_similar_shows(self, query: str, k: int = 3) -> List[str]:
        """Retrieve similar TV shows based on the query."""
        similar_docs = self.db.similarity_search(query, k=k)
        return [doc.page_content for doc in similar_docs]

class PromptLoader:
    @staticmethod
    def load_prompts(markdown_file: str) -> Dict[str, Dict[str, Any]]:
        """Load prompts and test queries from a markdown file."""
        with open(markdown_file, 'r') as f:
            content = f.read()
        
        sections = content.split('# Test Queries')
        prompts_section = sections[0]
        test_queries_section = sections[1] if len(sections) > 1 else ""
        
        prompts = {}
        prompt_blocks = re.findall(r'## (.*?)\n```prompt\n(.*?)\n```', prompts_section, re.DOTALL)
        for name, template in prompt_blocks:
            input_variables = re.findall(r'\{(\w+)\}', template)
            prompts[name.lower().replace(' ', '_')] = {
                "template": template.strip(),
                "input_variables": input_variables
            }
        
        return prompts

class PromptTester:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7, openai_api_key: str = None):
        self.llm = ChatOpenAI(
            temperature=temperature,
            model=model_name,
            openai_api_key=openai_api_key
        )
        self.rag = TVShowRAG(openai_api_key=openai_api_key)
        
    def get_recommendation(self, 
                         prompt_template: str, 
                         input_variables: List[str],
                         query: Dict[str, str]) -> str:
        """Get a recommendation using the prompt template and RAG."""
        # First, get similar shows from the database
        search_query = " ".join([f"{k}: {v}" for k, v in query.items()])
        similar_shows = self.rag.get_similar_shows(search_query)
        
        # Add the similar shows to the prompt template
        enhanced_template = f"""
        {prompt_template}
        
        Here are some similar TV shows from our database:
        {chr(10).join(similar_shows)}
        
        Please use this information to provide a more accurate recommendation.
        """
        
        prompt = PromptTemplate(
            input_variables=input_variables,
            template=enhanced_template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        try:
            response = chain.run(**query)
            return response
        except Exception as e:
            return f"Error: {str(e)}"

def main():
    st.set_page_config(page_title="TV Show Recommendation System", layout="wide")
    
    st.title("🎬 TV Show Recommendation System")
    st.markdown("""
    This system helps you find TV shows based on your preferences. 
    Choose a recommendation style and provide your preferences to get personalized recommendations.
    The recommendations are enhanced using our database of TV shows.
    """)
    
    # API Key Input
    st.sidebar.header("OpenAI API Settings")
    openai_api_key = st.sidebar.text_input(
        "Enter your OpenAI API Key",
        type="password",
        help="You can find your API key at https://platform.openai.com/account/api-keys"
    )
    
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key to continue.")
        st.stop()
    
    # Load prompts
    prompts = PromptLoader.load_prompts("prompts.md")
    
    # Sidebar for model settings
    with st.sidebar:
        st.header("Model Settings")
        model_name = st.selectbox(
            "Select Model",
            ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"],
            index=0
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make the output more creative, lower values make it more focused"
        )
    
    # Main content - single column layout
    st.header("Choose Recommendation Style")
    prompt_type = st.selectbox(
        "Select Recommendation Type",
        list(prompts.keys()),
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    selected_prompt = prompts[prompt_type]
    input_variables = selected_prompt["input_variables"]
    
    st.subheader("Your Preferences")
    query = {}
    for var in input_variables:
        if var == "preferences":
            query[var] = st.text_area(
                "What kind of TV shows do you like?",
                placeholder="e.g., I enjoy action-packed shows with strong female leads"
            )
        elif var == "genre":
            query[var] = st.selectbox(
                "Genre",
                ["Action", "Comedy", "Drama", "Science Fiction", "Thriller", "Horror", "Romance", "Documentary"]
            )
        elif var == "mood":
            query[var] = st.selectbox(
                "Mood",
                ["Excited", "Relaxed", "Thoughtful", "Inspired", "Scared", "Happy", "Sad"]
            )
        elif var == "length":
            query[var] = st.selectbox(
                "Preferred Length",
                ["Single season", "Multiple seasons", "Mini-series", "No preference"]
            )
        elif var == "favorite_show":
            query[var] = st.text_input(
                "Your Favorite TV Show",
                placeholder="e.g., Breaking Bad"
            )
        else:
            query[var] = st.text_input(
                var.replace('_', ' ').title(),
                placeholder=f"Enter your {var}"
            )
    
    if st.button("Get Recommendation"):
        if all(query.values()):
            tester = PromptTester(
                model_name=model_name,
                temperature=temperature,
                openai_api_key=openai_api_key
            )
            with st.spinner("Generating recommendation..."):
                recommendation = tester.get_recommendation(
                    prompt_template=selected_prompt["template"],
                    input_variables=input_variables,
                    query=query
                )
                st.markdown("### Your Personalized Recommendation")
                st.markdown(recommendation)
        else:
            st.warning("Please fill in all the required fields to get a recommendation.")

if __name__ == "__main__":
    main()

