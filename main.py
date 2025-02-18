import os


from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings


# Load environment variables from .env
load_dotenv()

#os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY")  # Use Groq API Key

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the books directory exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path."
        )

    # List all PDF files in the directory
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".pdf")]

    # Read the text content from each PDF file and store it with metadata
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = PyPDFLoader(file_path)  # Use PyPDFLoader for PDFs
        book_docs = loader.load()  # Extract text from PDF
        for doc in book_docs:
            # Add metadata to each document indicating its source
            doc.metadata = {"source": book_file}
            documents.append(doc)

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    filtered_documents = []
    for doc in documents:
        if "Contents" in doc.page_content or "Preface" in doc.page_content or "Introduction" in doc.page_content or "Introduction" in doc.page_content:
            continue  # Skip intro/table of contents
        filtered_documents.append(doc)

    docs = text_splitter.split_documents(filtered_documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # Or another Groq-supported model
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it
    print("\n--- Creating and persisting vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating and persisting vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Create a retriever for querying the vector store
# `search_type` specifies the type of search (e.g., similarity)
# `search_kwargs` contains additional arguments for the search (e.g., number of results to return)
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 40},
)

# query = "pseudoinstructions"
#
# relevant_docs = retriever.invoke(query)
#
# # Display the relevant results with metadata
# print("\n--- Relevant Documents ---")
# for i, doc in enumerate(relevant_docs, 1):
#     print(f"Document {i}:\n{doc.page_content}\n")
#     if doc.metadata:
#         print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

# Create a ChatOpenAI model
llm = ChatOpenAI(model = 'gpt-4o-mini')

# Contextualize question prompt
# This system prompt helps the AI understand that it should reformulate the question
# based on the chat history to make it a standalone question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever
# This uses the LLM to help reformulate the question based on chat history
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question prompt
# This system prompt helps the AI understand that it should provide concise answers
# based on the retrieved context and indicates what to do if the answer is unknown
# qa_system_prompt = (
#     "You are an assistant for question-answering tasks. Use "
#     "the following pieces of retrieved context to answer the "
#     "question. Only answer out of context as a last resort."
#     "If the context **contains the answer**, begin "
#     "your response with: 'This is from the document.'. If the "
#     "context **does not contain the answer**, start with: 'My "
#     "answer isn`t from your doc.'.Answer thoroughly, "
#     "in details if possible"
#     "\n\n"
#     "{context}"
# )

qa_system_prompt = (
    "You are an assistant for question-answering tasks. Your responses must be "
    "primarily based on the retrieved context provided below. Use it as your main "
    "source of information and extract as much relevant detail as possible."
    "\n\n"
    "If the context **contains enough information**, start your response with: "
    "'This is from the document.'. If the context **partially addresses the question** "
    "but is not fully sufficient, begin with: 'Based on the document, with additional details.'. "
    "In such cases, expand on the provided information using external knowledge to ensure the "
    "answer fully covers the question."
    "\n\n"
    "Only if the context is completely unrelated to the question, start with: "
    "'My answer isn’t from your doc, but here’s what I found.'."
    "\n\n"
    "Ensure responses are thorough and detailed."
    "\n\n"
    "{context}"
)


# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
# `create_stuff_documents_chain` feeds all retrieved context into the LLM
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

def translate_to_english(text):
    return GoogleTranslator(source="uk", target="en").translate(text)

def translate_to_ukrainian(text):
    return GoogleTranslator(source="en", target="uk").translate(text)

# Function to simulate a continual chat
def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []  # Collect chat history here (a sequence of messages)
    while True:
        query = input("Ви: ")
        if query.lower() == "exit":
            break

        translated_query = translate_to_english(query)

        # Retrieve relevant documents
        retrieved_docs = history_aware_retriever.invoke({"input": translated_query, "chat_history": chat_history})

        # If no relevant documents are found, say "I don't know."
        if not retrieved_docs:
            print("\nAI: I don't know.")
            continue

        # Print retrieved document chunks
        print("\n--- Retrieved Chunks ---")
        for i, doc in enumerate(retrieved_docs[:3]):
            print(f"Chunk {i + 1}: {doc.page_content}\n{'-' * 50}")

        # Process the query using the RAG chain
        result = rag_chain.invoke({"input": translated_query, "chat_history": chat_history})

        response = translate_to_ukrainian(result["answer"])
        print(f"\nAI: {response}")

        # Update chat history
        chat_history.append(HumanMessage(content=translated_query))
        chat_history.append(SystemMessage(content=result["answer"]))


# Main function to start the continual chat
if __name__ == "__main__":
    continual_chat()
#
