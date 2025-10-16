# Smart Second Brain Test Suite

This directory contains comprehensive tests for the Smart Second Brain project, including workflow components and API endpoints.

## Files

- `test_master_graph_builder.py` - Main test suite for MasterGraphBuilder (31 tests)
- `test_document_retriever.py` - Document retrieval tests (9 tests)
- `run_graph_tests.py` - Test runner script for easy execution
- `README.md` - This documentation file

## Test Coverage Summary

**Total Tests: 47**
- **API Tests**: 7 tests covering all Graph API endpoints
- **Service Tests**: 40 tests covering workflows and document retrieval
- **All Tests Pass**: 100% success rate with comprehensive mocking and real component testing

## Test Coverage

The test suite covers:

### 🔧 **API Endpoint Tests**
- Health endpoint functionality
- Text document ingestion (`/ingest`)
- PDF batch ingestion (`/ingest-pdfs`)
- Knowledge query processing (`/query`)
- Feedback submission (`/feedback`)
- Feedback status retrieval (`/feedback/{thread_id}`)
- Error handling and validation

### 🔧 **Initialization Tests**
- Constructor with various dependency combinations
- Mock dependency injection

### 🔀 **Node Method Tests**
- `input_router` - Routing between ingest and query workflows
- `chunk_doc_node` - Document chunking functionality
- `embed_node` - Embedding generation
- `store_node` - Vector storage operations
- `retriever_node` - Document retrieval
- `answer_gen_node` - LLM answer generation
- `human_review_node` - Human review workflow
- `validated_store_node` - Validated answer storage

### 🏗️ **Graph Structure Tests**
- Graph compilation
- Node and edge creation
- Workflow routing

### 🔄 **End-to-End Workflow Tests**
- Complete ingest workflow
- Complete query workflow
- Error handling and edge cases
- Real LLM integration tests
- Performance benchmarking

## Running Tests

### Option 1: Using the Test Runner (Recommended)

```bash
# Run all tests (unit + integration)
python tests/test_services/run_graph_tests.py --all

# Run only unit tests (mocked components)
python tests/test_services/run_graph_tests.py --unit

# Run only integration tests (real components)
python tests/test_services/run_graph_tests.py --integration

# Run all tests (default)
python tests/test_services/run_graph_tests.py

# Run a specific test
python tests/test_services/run_graph_tests.py --test test_initialization

# Run a specific test class
python tests/test_services/run_graph_tests.py --class TestMasterGraphBuilder
```

### Option 2: Using pytest directly

```bash
# Run all tests (47 tests total)
pytest tests/ -v

# Run API tests only (7 tests)
pytest tests/test_api/ -v

# Run service tests only (40 tests)
pytest tests/test_services/ -v

# Run with coverage
pytest tests/ --cov=api --cov=agentic --cov-report=html

# Run specific test file
pytest tests/test_services/test_master_graph_builder.py -v

# Run specific test
pytest tests/test_services/test_master_graph_builder.py::TestMasterGraphBuilder::test_initialization -v
```

### Option 3: From project root

```bash
# Run all tests
pytest tests/ -v

# Run with detailed output
pytest tests/ -v --tb=long

# Run specific test class
pytest tests/test_api/test_graph_api.py::TestGraphAPI -v
```

## Test Types

### 🎭 **Unit Tests (Mocked Components)**
- **LLM Client**: Mocked OpenAI/ChatOpenAI responses
- **Document Retriever**: Mocked retriever with sample documents
- **Vector Store**: Mocked vector database operations
- **Fast execution**: No external API calls
- **Reliable**: No network dependencies

### 🤖 **Integration Tests (Real Components)**
- **Real OpenAI LLM**: Uses actual GPT-4o-mini model (OpenAI or Azure OpenAI)
- **Real Vector Store**: Chroma with OpenAI embeddings
- **Real Text Splitting**: RecursiveCharacterTextSplitter
- **Performance Benchmarking**: Timing measurements
- **End-to-End Workflows**: Complete document ingestion and querying
- **Automatic Provider Detection**: Uses Azure OpenAI if endpoint configured, otherwise OpenAI

## Test Dependencies

### 📊 **Test Data**
- **Sample Documents**: Multi-paragraph test documents
- **Sample Queries**: Realistic user queries
- **Knowledge States**: Complete state objects for testing

### 🔑 **Required Environment Variables**
```bash
# For integration tests - add to your .env file
OPENAI_API_KEY=your-openai-api-key-here

# Optional: For Azure OpenAI
AZURE_OPENAI_ENDPOINT_URL=https://your-resource-name.openai.azure.com/
```

### 📦 **Required Packages for Integration Tests**
```bash
pip install langchain-openai langchain-community
```

## Test Structure

```python
class TestMasterGraphBuilder:
    # Fixtures for test setup
    @pytest.fixture
    def mock_llm(self): ...
    
    @pytest.fixture
    def mock_retriever(self): ...
    
    @pytest.fixture
    def graph_builder(self): ...
    
    # Individual test methods
    def test_initialization(self): ...
    def test_input_router_ingest(self): ...
    def test_chunk_doc_node(self): ...
    # ... more tests
```

## Expected Test Results

### Test Results
When all tests pass, you should see:

```
============================= test session starts =============================
platform darwin -- Python 3.12.2, pytest-8.4.1, pluggy-1.6.0
collecting ... collected 47 items

tests/test_api/test_graph_api.py::TestGraphAPI::test_health_endpoint PASSED [  2%]
tests/test_api/test_graph_api.py::TestGraphAPI::test_ingest_endpoint_success PASSED [  4%]
tests/test_api/test_graph_api.py::TestGraphAPI::test_ingest_endpoint_missing_document PASSED [  6%]
tests/test_api/test_graph_api.py::TestGraphAPI::test_ingest_endpoint_empty_document PASSED [  8%]
tests/test_api/test_graph_api.py::TestGraphAPI::test_ingest_endpoint_workflow_error PASSED [ 10%]
tests/test_api/test_graph_api.py::TestGraphAPI::test_ingest_endpoint_minimal_data PASSED [ 12%]
tests/test_api/test_graph_api.py::TestGraphAPI::test_ingest_endpoint_with_metadata PASSED [ 14%]
tests/test_services/test_document_retriever.py::TestSmartDocumentRetriever::test_retriever_initialization PASSED [ 17%]
...
tests/test_services/test_master_graph_builder.py::TestMasterGraphBuilder::test_real_performance_benchmark PASSED [100%]

======================== 47 passed, 30 warnings in 51.04s =======================
```

### Coverage Report
```
_______________ coverage: platform darwin, python 3.12.2-final-0 _______________
Name                         Stmts   Miss  Cover   Missing
----------------------------------------------------------
api/__init__.py                  3      0   100%
api/core/__init__.py             0      0   100%
api/core/config.py              25      0   100%
api/main.py                     40     23    42%   28-74, 120, 132-134, 151, 170
api/routes/__init__.py           0      0   100%
api/routes/v1/__init__.py        0      0   100%
api/routes/v1/graph_api.py     346    200    42%   114, 174-176, 524-551, 615-616, 679-730, 781, 783, 796-799, 850-967, 1001-1138, 1166-1255
----------------------------------------------------------
TOTAL                          414    223    46%
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the project root
2. **Missing Dependencies**: Install pytest: `pip install pytest`
3. **Path Issues**: Use the test runner script for automatic path setup

### Debug Mode

For detailed debugging, run with:

```bash
pytest tests/test_services/test_master_graph_builder.py -v -s --tb=long
```

## Contributing

When adding new features to the Smart Second Brain project:

1. **Add corresponding tests** for new API endpoints and workflow methods
2. **Update existing tests** if changing behavior
3. **Test edge cases** and error conditions
4. **Maintain test coverage** above 40% (current: 46%)
5. **Follow the existing test patterns** for mocking and assertions

## Integration with CI/CD

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions step
- name: Run Smart Second Brain Tests
  run: |
    pytest tests/ -v --cov=api --cov=agentic --cov-report=xml
    
- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```
