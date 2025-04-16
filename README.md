# TV Show Recommendation System

A Streamlit-based application that provides personalized TV show recommendations using OpenAI's language models and a RAG (Retrieval Augmented Generation) system.

## Features

- Interactive UI for TV show recommendations
- Support for multiple recommendation styles
- RAG system using TV show data
- Customizable model settings
- Secure API key management

## Try it Online

You can test the application at: https://recommendmetv.streamlit.app/

## Components

### TVShowRAG Class
- Handles the retrieval-augmented generation system
- Loads and processes TV show data from CSV
- Creates embeddings and vector store using FAISS
- Retrieves similar shows based on user queries

### PromptLoader Class
- Loads and parses prompt templates from markdown files
- Extracts input variables from templates
- Organizes prompts for different recommendation styles

### PromptTester Class
- Manages the language model interactions
- Combines user preferences with RAG results
- Generates personalized recommendations
- Handles error cases gracefully

## Usage

1. Install required packages:
```bash
pip install streamlit langchain langchain-openai langchain-community faiss-cpu python-dotenv
```

2. Run the application:
```bash
streamlit run app.py
```

3. Enter your OpenAI API key in the sidebar

4. Select your preferences:
   - Choose a recommendation style
   - Select model and temperature settings
   - Input your TV show preferences
   - Get personalized recommendations

## File Structure

- `app.py`: Main application file
- `prompts.md`: Contains prompt templates for different recommendation styles
- `TV_show_data_summary_only.csv`: TV show database
- `.env`: (Optional) For storing API keys

## Requirements

- Python 3.8+
- OpenAI API key
- Required Python packages (see Usage section)

## Note

Remember to keep your OpenAI API key secure and never share it publicly.