# AI Chat Summarizer

A professional Flask-based web application that leverages OpenAI's GPT-4 to generate structured, investor-ready summaries of business meetings and conversations.

## Features

- **Meeting Summarization**: Convert meeting notes and conversations into concise, actionable summaries
- **Multi-Format File Support**: Upload and analyze PDF, DOCX, DOC, and TXT documents
- **Structured Output**: Generates summaries with organized sections:
  - Summary (key decisions and outcomes)
  - Action Points (specific tasks with assignments and deadlines)
  - Dates & Deadlines
  - Document Summary
  - Detailed Summary
  - Confidence Assessment

- **Chat History Management**: 
  - Continue previous conversations
  - Save and export meeting summaries
  - View chat history with timestamps
  - Delete old conversations

- **Professional Formatting**: Business-ready outputs suitable for executives and investors
- **Error Handling**: Robust error handling with comprehensive logging
- **File Management**: Automatic cleanup of old uploaded files

## Project Structure

```
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/            # HTML templates
│   ├── home.html         # Main interface
│   ├── result.html       # Summary results display
│   ├── 404.html          # 404 error page
│   ├── 500.html          # 500 error page
│   └── error.html        # Generic error page
├── static/               # Static assets
│   ├── styles.css        # Application styling
│   └── scripts.js        # Client-side JavaScript
├── chat_history/         # Stored conversation histories (JSON)
├── saved_conversations/  # Exported conversations (JSON)
└── uploads/             # Temporary file uploads
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SauravChauhan666/AI-Chat-Summarizer.git
   cd AI-Chat-Summarizer
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   FLASK_SECRET_KEY=your_secret_key_here
   FLASK_ENV=development  # or production
   HOST=0.0.0.0
   PORT=5000
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

   The application will be available at `http://localhost:5000`

## Usage

### Creating a New Summary

1. Navigate to the home page
2. Select "New Conversation" 
3. Enter your meeting notes or conversation content
4. Optionally upload a supporting document (PDF, DOCX, DOC, or TXT)
5. Click "Generate Summary"
6. View the structured summary with all sections

### Continuing Previous Conversations

1. From the home page, select an existing conversation from the list
2. Add new meeting notes
3. The AI will maintain context from previous messages
4. Generate an updated summary

### Saving Conversations

1. After generating a summary, enter a name for the conversation
2. Click "Save Conversation"
3. Conversation is stored in the `saved_conversations/` folder with a timestamp

### Exporting Data

- Use the export feature to download chat data as JSON
- Saved conversations are automatically stored as JSON files
- All data is properly formatted for further processing or archival

## Configuration

### Environment Variables

- `OPENAI_API_KEY` (Required): Your OpenAI API key for GPT-4 access
- `FLASK_SECRET_KEY`: Secret key for Flask session management
- `FLASK_ENV`: Set to 'development' or 'production'
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 5000)

### Application Settings

- **Max File Size**: 25 MB
- **Allowed File Types**: .pdf, .docx, .doc, .txt
- **Conversation Context**: Last 10 messages used for context
- **Auto-cleanup**: Files older than 30 days are automatically removed

## API Endpoints

### Summary Generation
- `POST /` - Generate a new summary from meeting notes

### Chat Management
- `GET /continue/<chat_id>` - Continue an existing conversation
- `POST /save_conversation` - Save a conversation with a custom name
- `POST /delete_chat/<chat_id>` - Delete a conversation

### API Routes
- `GET /api/chat/<chat_id>` - Get chat data in JSON format
- `GET /api/chats` - List all available chats
- `GET /health` - Health check endpoint
- `GET /export_chat/<chat_id>` - Export chat as JSON download

### Error Handlers
- `404` - Page not found
- `413` - File too large
- `500` - Internal server error

## Dependencies

- **Flask**: Web framework
- **OpenAI**: GPT-4 API client
- **python-dotenv**: Environment variable management
- **Werkzeug**: WSGI utilities and security
- **PyMuPDF (fitz)**: PDF document processing
- **python-docx**: DOCX document processing

## Logging

Application logs are written to `app.log` and displayed in the console. Logging includes:
- File operations
- API calls and responses
- Error tracking
- Performance metrics



## Error Handling

The application includes comprehensive error handling:
- File validation (size, type, format)
- API timeout and retry logic (max 3 retries)
- Graceful degradation with user-friendly error messages
- Automatic backup creation before saving
- Detailed logging for debugging

## Performance Considerations

- **Token Limiting**: Context limited to last 10 messages to prevent token overflow
- **File Size Limit**: Maximum 25 MB per upload
- **Response Caching**: Conversation history cached locally
- **Cleanup Task**: Automatic removal of files older than 30 days

## Troubleshooting

### OpenAI API Connection Issues
- Verify `OPENAI_API_KEY` is set correctly
- Check API quota and billing status
- Review `app.log` for detailed error messages

### File Upload Errors
- Ensure file size is under 25 MB
- Check that file format is supported (.pdf, .docx, .doc, .txt)
- Verify sufficient disk space in `uploads/` folder

### Summary Generation Issues
- Check OpenAI API status and rate limits
- Ensure sufficient token availability
- Review conversation context isn't exceeding limits



