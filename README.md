# AI Agent Challenge - Bank Statement Parser

An intelligent AI agent that automatically generates custom parsers for bank statement PDFs using Gemini AI models. This agent can analyze bank statement formats and create tailored Python parsers to extract transaction data into structured CSV formats.

## Features

- **Automated Parser Generation**: Uses Gemini AI to create custom parsers for different bank statement formats
- **Multi-Bank Support**: Currently supports SBI and ICICI bank statements with extensible architecture
- **PDF to CSV Conversion**: Extracts transaction data from PDF statements into structured CSV format
- **Self-Correcting AI**: Automatically tests and refines generated parsers for accuracy
- **Flexible AI Models**: Compatible with Gemini 2.5 Flash, Gemini Pro, and other Google AI models

## Prerequisites

- Python 3.8 or higher
- Google AI Studio API key
- Virtual environment (recommended)

## Installation & Setup

### 1. Clone the Repository
```bash
git clone <https://github.com/anjanabharamnaikar/ai-agent-challenge-karbon
cd ai-agent-challenge-karbon
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory:

```bash
touch .env
```

Add your Google AI Studio API key to the `.env` file:

```env
GOOGLE_API_KEY=your_google_ai_studio_api_key_here
```

**Note**: You can use any Gemini model (2.5 Flash, Pro, etc.) by modifying the model parameter in `agent.py`. The default is set to `gemini-2.5-flash`.

### 5. Get Google AI Studio API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and paste it into your `.env` file

## How to Use

### Running the AI Agent

The agent can generate parsers for different bank statement formats. Currently supported banks:

- **SBI** (State Bank of India)
- **ICICI** (ICICI Bank)

### Basic Usage

```bash
# Generate parser for SBI statements
python agent.py --target sbi

# Generate parser for ICICI statements
python agent.py --target icici
```

### Expected Output Structure

After running the agent, you'll find:
- Generated parser: `custom_parsers/[bank_name]_parser.py`
- Sample data: `data/[bank_name]/[bank_name]_sample.pdf` and `[bank_name]_sample.csv`

## Project Structure

```
ai-agent-challenge-karbon/
├── agent.py                 # Main AI agent orchestrator
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── .env                    # Environment variables (create this)
├── .gitignore             # Git ignore rules
├── custom_parsers/        # Generated parser modules
│   └── [bank_name]_parser.py
├── data/                  # Sample bank statement data
│   ├── sbi/
│   │   ├── sbi_sample.pdf
│   │   └── sbi_sample.csv
│   └── icici/
│       ├── icici_sample.pdf
│       └── icici_sample.csv
└── tests/
    └── test_parser.py     # Parser testing utilities
```

## Understanding the AI Agent Workflow

1. **Planning Phase**: Analyzes the target CSV schema and creates a parsing strategy
2. **Code Generation**: Uses Gemini AI to generate Python parser code
3. **Testing Phase**: Tests the generated parser against sample data
4. **Self-Correction**: Refines the parser if tests fail (up to 3 attempts)

## Customization

### Using Different Gemini Models

Edit the model parameter in `agent.py`:

```python
# Change this line to use different models:
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)

# Available models:
# - gemini-2.5-flash (default)
# - gemini-pro
# - gemini-pro-vision
```

### Adding New Bank Support

1. Add sample PDF and CSV files to `data/[new_bank]/`
2. Run: `python agent.py --target [new_bank]`
3. The agent will generate a new parser automatically

## Testing

The agent includes built-in testing that:
- Validates parser output against expected CSV format
- Checks data type consistency
- Ensures transaction amounts are correctly signed (debits negative, credits positive)

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your `.env` file has the correct `GOOGLE_API_KEY`
2. **PDF Parsing Issues**: Check if the PDF is a scanned document (may require OCR)
3. **Model Access**: Verify your Google AI Studio account has access to the selected model

### Debug Mode

Enable verbose logging by modifying the agent:
```python
# In agent.py, change:
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, verbose=True)
```

## Contributing

Feel free to:
- Add support for new bank statement formats
- Improve parser accuracy
- Enhance the AI agent's planning capabilities
- Add additional validation checks

## License

This project is part of the AI Agent Challenge. Please check the repository for specific licensing information.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Ensure your `.env` file is properly configured
3. Verify your Google AI Studio API key is active

