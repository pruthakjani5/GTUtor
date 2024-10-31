# GTUtor: Dynamic Multi-Subject Chat System

GTUtor is an intelligent tutoring system specifically designed for Gujarat Technological University (GTU) students. It combines the power of Google's Gemini Pro AI with a document-based knowledge system to provide accurate, context-aware responses across multiple subjects.

## Features

- 📚 **Multi-Subject Support**: Create and manage multiple subjects with independent knowledge bases
- 📑 **Document Integration**: Upload PDF documents or provide URLs to enhance the knowledge base
- 💬 **Intelligent Chat Interface**: Dynamic conversation system with history tracking
- 🔍 **Context-Aware Responses**: Combines document knowledge with Gemini Pro's capabilities
- 📊 **Database Management**: Built-in tools to manage document databases for each subject
- 🎓 **GTU-Focused**: Specifically tailored for GTU curriculum and courses
- 💾 **Persistent Storage**: Automatically saves chat histories and subject data
- 📋 **Copy Functionality**: Easy copying of questions and answers

## Installation

1. Clone the repository:
```bash
git clone https://github.com/pruthakjani5/GTUtor.git
cd gtutor
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

If you do need a fresh setup without the trial database and previous chats then delete the "gtutor_data" folder, running app.py will create a new one.
## Required Dependencies

- streamlit
- requests
- pypdf
- google-generativeai
- chromadb
- python-dotenv
- tempfile
- markdown
- clipboard

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Access the application through your web browser (typically at `http://localhost:8501`)

3. Select or create a subject from the dropdown menu

4. Upload PDF documents or provide PDF URLs to build the subject's knowledge base

5. Start asking questions in the chat interface

## Features in Detail

### Subject Management
- Create new subjects
- Delete existing subjects
- Clear subject databases
- Track document count per subject

### Document Management
- Upload PDF files (up to 10MB)
- Add documents via URL
- Automatic text extraction and chunking
- Persistent storage of document data

### Chat Interface
- Real-time question answering
- Chat history tracking
- Copy questions and answers
- Delete individual messages
- Start new conversations
- Enhanced UI with user/bot avatars

### Answer Generation
- Context-aware responses using uploaded documents
- Fallback to Gemini Pro's knowledge when needed
- Structured and formatted responses
- Academic tone with GTU curriculum focus

## Project Structure

```
gtutor/
├── app.py                 # Main application file
├── .env                   # Environment variables
├── requirements.txt       # Project dependencies
└── gtutor_data/          # Data directory
    ├── dbs/              # Subject databases
    ├── chat_histories/   # Conversation histories
    └── subjects.json     # Subject list
```

## Technical Implementation

- **Document Processing**: Uses `pypdf` for PDF text extraction with automatic chunking
- **Vector Database**: Implements `chromadb` for efficient text storage and retrieval
- **UI Framework**: Built with `streamlit` for responsive web interface
- **AI Integration**: Utilizes Google's Gemini Pro API for intelligent responses
- **Data Persistence**: JSON-based storage for chat histories and subject data
- **Markdown Support**: Enhanced text formatting for responses

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Generative AI for the Gemini Pro API
- Streamlit for the web framework
- ChromaDB for the vector database implementation

## Support

For support and questions, please open an issue in the GitHub repository or contact the maintainers.
