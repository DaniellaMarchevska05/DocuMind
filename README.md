
# DocuMind 📚🤖
**AI-powered study companion to help busy students pass their exams using official tutor-published documentation as the main source of information.**

## 📌 Overview  
DocuMind is an AI-driven study assistant designed to support students by leveraging official course materials provided by tutors. It extracts, organizes, and retrieves relevant information from academic resources, enabling students to easily find answers to their questions, ensuring that the answers are based solely on the tutor-published documents.

## 🚀 Features  
- **Document-Based Q&A:** Answers questions using tutor-provided PDFs as the sole knowledge source.  
- **Context-Aware Retrieval:** Intelligent AI-powered search that fetches the most relevant sections of documents.  
- **Chat History Awareness:** Reformulates questions based on previous interactions for accurate and consistent answers.  
- **Efficient Information Retrieval:** Utilizes ChromaDB for quick, seamless document indexing and retrieval.  
- **Multiple AI Models Support:** Uses OpenAI, Groq, and Hugging Face embeddings for optimal AI performance.  
- **Persistent Storage:** Maintains document processing across sessions using a persistent vector store 

## 🔧 Installation  
1. **Clone the repository:**  
   ```bash
   git clone https://github.com/yourusername/documind.git
   cd documind
   ```
2. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables:**  
   Create a `.env` file and add your API keys (OpenAI, Groq, etc.):  
   ```
   OPENAI_API_KEY=your_openai_key
   ```
4. **Add PDFs in the `books/` directory**  
   Place your tutor-provided documents in the `books/` folder.  

5. **Run the application:**  
   ```bash
   python documind.py
   ```

## 🛠️ How It Works  
1. **Load and process PDFs** from the `books/` directory, extracting text content and metadata.  
2. **Create a vector-based database** for fast document retrieval.  
3. **Answer user questions** by retrieving the most relevant documents from the database.  
4. **Refine answers** using AI-powered context-aware retrieval that incorporates chat history.

## 🎯 Usage Example  
- **User:** *What is the main theme of Chapter 2?*  
- **AI:** *This is from the document: Chapter 2 covers the theory of relativity and its applications...*  
- **User:** *Can you explain it in simpler terms?*  
- **AI:** *Based on the document, with additional details... it refers to the way objects in motion experience time differently from those at rest.*

## 📌 Future Enhancements  
- A user-friendly UI for better interaction.    
