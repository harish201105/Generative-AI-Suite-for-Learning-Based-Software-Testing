# 🧠 Advanced Python Complexity Analyzer & AI Model Benchmarking

A comprehensive tool for analyzing Python code complexity and benchmarking AI model performance across multiple providers. This application combines advanced static code analysis with intelligent test case generation using state-of-the-art language models.

## 🚀 Features

### Core Functionality
- **Multi-Model AI Integration**: Support for 10+ AI models from leading providers
- **Advanced Complexity Analysis**: Comprehensive Python code complexity metrics
- **Intelligent Test Case Generation**: AI-powered test case creation with validation
- **Performance Benchmarking**: Real-time comparison of model performance metrics
- **Interactive Visualizations**: Rich charts and graphs for data analysis
- **Robust Error Handling**: Advanced JSON parsing with Python expression evaluation

### Supported AI Models
- **OpenAI**: GPT-5, GPT-4o
- **Anthropic**: Claude 4 Sonnet
- **Google**: Gemini 2.5 Flash
- **Mistral AI**: Mistral Large
- **Meta**: Llama 4 Maverick 
- **Microsoft**: Phi-4
- **Alibaba**: Qwen3
- **DeepSeek**: DeepSeek V3
- **xAI**: Grok-3
- **Cohere**: Command R+

### Analysis Capabilities
- **Cyclomatic Complexity**: McCabe complexity analysis
- **Cognitive Complexity**: Human readability metrics
- **Halstead Metrics**: Software science measurements
- **Maintainability Index**: Code maintainability scoring
- **SLOC Analysis**: Source lines of code counting
- **Dependency Analysis**: Import and module dependencies
- **Performance Profiling**: Execution time and resource usage

## 📋 Requirements

- **Python**: 3.8+ (recommended: 3.10+)
- **Operating System**: macOS, Linux, Windows
- **Memory**: 4GB+ RAM (8GB+ recommended for large codebases)
- **API Keys**: Required for AI model providers (see setup section)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd LBST
```

### 2. Create Virtual Environment
```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the root directory with your API keys:

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic (Claude)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google (Gemini)
GOOGLE_API_KEY=your_google_api_key_here

# Mistral AI
MISTRAL_API_KEY=your_mistral_api_key_here

# Groq (for Llama models)
GROQ_API_KEY=your_groq_api_key_here

# Azure (for Phi models)
AZURE_AI_INFERENCE_API_KEY=your_azure_api_key_here
AZURE_AI_INFERENCE_ENDPOINT=your_azure_endpoint_here

# Additional providers (optional)
COHERE_API_KEY=your_cohere_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
XAI_API_KEY=your_xai_api_key_here
```

## 🚀 Usage

### Starting the Application
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Basic Workflow

1. **Code Input**: Paste or type your Python code in the text area
2. **Model Selection**: Choose which AI models to benchmark
3. **Analysis Type**: Select complexity analysis and/or test generation
4. **Generate Results**: Click "Analyze & Generate" to start processing
5. **Review Output**: Examine complexity metrics, generated tests, and performance comparisons

### Advanced Features

#### Complexity Analysis Modes
- **Quick Analysis**: Basic metrics (cyclomatic complexity, SLOC)
- **Comprehensive Analysis**: Full suite including Halstead metrics and maintainability index
- **Custom Analysis**: Select specific metrics to calculate

#### Test Generation Options
- **Unit Tests**: Generate pytest-compatible unit tests
- **Integration Tests**: Create broader integration test scenarios
- **Edge Case Tests**: Focus on boundary conditions and error cases
- **Performance Tests**: Generate benchmarking and load tests

#### Benchmarking Configurations
- **Response Time**: Measure API response latency
- **Token Usage**: Track input/output token consumption
- **Success Rate**: Monitor parsing and generation success
- **Quality Metrics**: Evaluate test case completeness and accuracy

## 📊 Output Formats

### Complexity Metrics
```json
{
  "cyclomatic_complexity": 8,
  "cognitive_complexity": 12,
  "halstead_difficulty": 15.2,
  "maintainability_index": 68.4,
  "lines_of_code": 145,
  "function_count": 6,
  "class_count": 2
}
```

### Performance Benchmarks
```json
{
  "model": "GPT-5",
  "response_time": 2.34,
  "tokens_used": 1250,
  "success_rate": 98.5,
  "test_cases_generated": 12,
  "quality_score": 8.7
}
```

## 🏗️ Architecture

### Project Structure
```
LBST/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables
├── .gitignore                     # Git ignore rules
├── README.md                      # This file
├── api/
│   ├── models/                    # AI model implementations
│   │   ├── base_model.py          # Abstract base class
│   │   ├── gpt_model.py           # OpenAI GPT models
│   │   ├── claude_model.py        # Anthropic Claude
│   │   ├── gemini_model.py        # Google Gemini
│   │   ├── mistral_model.py       # Mistral AI models
│   │   ├── llama_model.py         # Meta Llama models
│   │   ├── phi_model.py           # Microsoft Phi models
│   │   ├── qwen_model.py          # Alibaba Qwen models
│   │   ├── deepseek_v3_model.py   # DeepSeek models
│   │   ├── grok3_model.py         # xAI Grok models
│   │   └── cohere_model.py        # Cohere models
│   ├── analysis/                  # Analysis engines
│   │   ├── complexity_analyzer.py # Code complexity analysis
│   │   └── performance_analyzer.py # Model performance tracking
│   └── utils/                     # Utility functions
│       └── json_parser.py         # Enhanced JSON parsing
```

### Key Components

#### Model Architecture
- **Base Model**: Abstract interface for all AI providers
- **Provider Models**: Specific implementations for each AI service
- **Error Handling**: Robust fallback mechanisms and retry logic
- **Rate Limiting**: Built-in respect for API rate limits

#### Analysis Engine
- **Static Analysis**: AST-based code parsing and metrics calculation
- **Dynamic Analysis**: Runtime performance and behavior assessment
- **Comparative Analysis**: Cross-model performance evaluation

#### JSON Parser
- **Python Expression Handling**: Safe evaluation of Python code in JSON
- **Regex Patterns**: Advanced pattern matching for complex expressions
- **Error Recovery**: Graceful handling of malformed responses

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest

# Run specific test files
python -m pytest test_complexity_analyzer.py
python -m pytest test_enhanced_analyzer.py
python -m pytest test_multi_language_analyzer.py

# Run with coverage
python -m pytest --cov=api
```

### Benchmarking
```bash
# Run performance benchmarks
python run_complexity_benchmark.py

# Generate detailed performance reports
python -c "from api.analysis.complexity_benchmarker import ComplexityBenchmarker; ComplexityBenchmarker().run_comprehensive_benchmark()"
```

## 🛡️ Security & Privacy

- **API Key Protection**: Environment variables for secure credential storage
- **No Data Logging**: Code snippets are not stored or logged
- **Local Processing**: Complexity analysis runs entirely locally
- **Secure Transmission**: HTTPS for all API communications

## 🔧 Troubleshooting

### Common Issues

#### Missing API Keys
**Error**: `Configuration Error: Missing required API key`
**Solution**: Ensure all required API keys are set in your `.env` file

#### Model Timeout
**Error**: `Request timeout after 30 seconds`
**Solution**: Try a different model or simplify your code snippet

#### JSON Parsing Errors
**Error**: `All parsing strategies failed`
**Solution**: The enhanced JSON parser should handle most cases; report persistent issues

#### Package Installation Issues
**Error**: `No module named 'streamlit'`
**Solution**: Ensure virtual environment is activated and run `pip install -r requirements.txt`

### Performance Optimization

- **Model Selection**: Start with faster models (Phi, Qwen) for initial testing
- **Code Size**: Break large files into smaller functions for analysis
- **Concurrent Analysis**: Use multiple models simultaneously for faster results
- **Caching**: Results are cached within sessions to avoid redundant API calls

## 🤝 Contributing

1. **Fork the Repository**: Create your own fork on GitHub
2. **Create Feature Branch**: `git checkout -b feature/amazing-feature`
3. **Make Changes**: Implement your improvements
4. **Add Tests**: Ensure new features have test coverage
5. **Commit Changes**: `git commit -m 'Add amazing feature'`
6. **Push Branch**: `git push origin feature/amazing-feature`
7. **Open Pull Request**: Submit your changes for review

### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints for new functions
- Include docstrings for public methods
- Write tests for new functionality
- Update README for significant changes

## 🙏 Acknowledgments

- **AST Module**: Python's Abstract Syntax Tree for code analysis
- **Streamlit**: For the beautiful web interface
- **AI Providers**: OpenAI, Anthropic, Google, Mistral, Meta, Microsoft, Alibaba, DeepSeek, xAI, Cohere
- **Open Source Community**: For the libraries and tools that make this possible

**Last Updated**: September 23, 2025
**Version**: 2.0.0
**Python Compatibility**: 3.8+
