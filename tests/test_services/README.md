# MasterGraphBuilder Tests

This directory contains comprehensive tests for the MasterGraphBuilder workflow component of the Smart Second Brain project.

## Files

- `test_master_graph_builder.py` - Main test suite for MasterGraphBuilder
- `run_graph_tests.py` - Test runner script for easy execution
- `README.md` - This documentation file

## Test Coverage

The test suite covers:

### 🔧 **Initialization Tests**
- Constructor with various dependency combinations
- Mock dependency injection

### 🔀 **Node Method Tests**
- `input_router` - Routing between ingest and query workflows
- `chunk_doc_node` - Document chunking functionality
- `embed_node` - Embedding generation (placeholder)
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
# Run all tests
pytest tests/test_services/test_master_graph_builder.py -v

# Run with coverage
pytest tests/test_services/test_master_graph_builder.py --cov=agentic.workflows.master_graph_builder --cov-report=html

# Run specific test
pytest tests/test_services/test_master_graph_builder.py::TestMasterGraphBuilder::test_initialization -v
```

### Option 3: From project root

```bash
# Run all service tests
pytest tests/test_services/ -v

# Run with detailed output
pytest tests/test_services/ -v --tb=long
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

### Unit Tests
When unit tests pass, you should see:

```
🧪 Running Unit Tests (Mocked Components)
============================================================
test_master_graph_builder.py::TestMasterGraphBuilder::test_initialization PASSED
test_master_graph_builder.py::TestMasterGraphBuilder::test_input_router_ingest PASSED
test_master_graph_builder.py::TestMasterGraphBuilder::test_input_router_query PASSED
...
✅ All unit tests passed!
```

### Integration Tests
When integration tests pass, you should see:

```
🧪 Running Integration Tests (Real Components)
============================================================
test_master_graph_builder.py::TestMasterGraphBuilder::test_real_llm_answer_generation PASSED
🤖 Real LLM Response: Based on the context provided...

test_master_graph_builder.py::TestMasterGraphBuilder::test_real_document_ingestion_with_vectorstore PASSED
✅ Document 1 ingested successfully
✅ Document 2 ingested successfully
🔍 Retrieved 2 documents
🤖 Generated Answer: Knowledge graphs are powerful tools...

test_master_graph_builder.py::TestMasterGraphBuilder::test_real_performance_benchmark PASSED
⏱️  Ingestion time for 3 documents: 2.34 seconds
⏱️  Query time: 1.87 seconds
📊 Performance Summary:
   - Ingestion: 2.34s for 3 docs
   - Query: 1.87s
   - Total: 4.21s
✅ All integration tests passed!
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

When adding new features to MasterGraphBuilder:

1. **Add corresponding tests** for new node methods
2. **Update existing tests** if changing behavior
3. **Test edge cases** and error conditions
4. **Maintain test coverage** above 90%

## Integration with CI/CD

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions step
- name: Run MasterGraphBuilder Tests
  run: |
    python tests/test_services/run_graph_tests.py
```
