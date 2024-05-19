import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os
from transformers import BertTokenizer, BertModel
from bert_score import BERTScorer
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings




def get_vectorstore_from_url(url):
    if os.path.exists("store/un_sdg_chroma_cosine"):
        vector_store = Chroma(persist_directory="store/un_sdg_chroma_cosine", embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))
        return vector_store
    # get the text in document form
    else: 
        loader = WebBaseLoader(url)
        document = loader.load()
        
        # split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(document)
        
        # create a vectorstore from the chunks
        vector_store = Chroma.from_documents(document_chunks, SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),collection_metadata={"hnsw:space": "cosine"}, persist_directory="store/un_sdg_chroma_cosine")

        return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatGroq(groq_api_key='',
             model_name="mixtral-8x7b-32768") 
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    
    llm = ChatGroq(groq_api_key='',
             model_name="mixtral-8x7b-32768")
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "act as a senior customer care excutive and help useres soughting out their quries working noise coumpeny be polite and friendly  Answer the user's questions based on the below context:\n\n{context} make sure to provide all the details, if the answer is not in the provided context just say, 'answer is not available in the context', don't provide the wrong answer make sure if the person asks any any externl recomandation only provide information related to noise coumpany only , if user askas you anuthing other than noise coumpany just say 'sorry i can't help you with that'"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

# app config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Noise Support Chatbot")

def score(text1, text2):
    # BERTScore calculation
    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = scorer.score([text1], [text2])
    return f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}"


if __name__ == "__main__":
    # session state
    if os.path.exists("store/un_sdg_chroma_cosine"):
        st.session_state.vector_store = Chroma(persist_directory="store/un_sdg_chroma_cosine", embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))



    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url('https')    

    # user input
    user_query = st.chat_input("Type your message here...")

    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        docs = st.session_state.vector_store .similarity_search(user_query)
        t = ''
        for doc in docs:
            t += doc.page_content
        print(score(response,t))
        
       

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
