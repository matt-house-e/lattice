# Contributing to Lattice

Thank you for your interest in contributing to Lattice! This document provides guidelines and information for contributors.

## Development Setup

### Prerequisites
- Python 3.9 or higher
- Git

### Setting Up the Development Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/lattice.git
   cd lattice
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## Development Workflow

### Branch Naming Convention
- `feature/description` - New features
- `fix/description` - Bug fixes  
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test improvements

### Making Changes

1. **Create a feature branch:**
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write code following the existing style
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests and checks:**
   ```bash
   # Run tests
   python -m pytest tests/
   
   # Run code formatting
   black lattice tests
   isort lattice tests
   
   # Run type checking
   mypy lattice
   ```

4. **Commit your changes:**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

### Commit Message Convention

We use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Examples:
```bash
git commit -m "feat: add async processing support"
git commit -m "fix: handle empty CSV files gracefully" 
git commit -m "docs: update API documentation"
```

### Pull Request Process

1. **Push your branch:**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request:**
   - Use a clear, descriptive title
   - Fill out the PR template completely
   - Link any related issues
   - Request review from maintainers

3. **Address feedback:**
   - Make requested changes
   - Push additional commits
   - Respond to review comments

## Code Style Guidelines

### Python Code Style
- Follow PEP 8
- Use Black for code formatting (line length: 100)
- Use isort for import sorting
- Add type hints for all public functions
- Use meaningful variable and function names

### Documentation Style
- Use docstrings for all public functions and classes
- Follow Google-style docstrings
- Include examples in docstrings where helpful
- Keep README and docs up to date

### Testing Guidelines
- Write tests for all new functionality
- Use pytest for testing framework
- Aim for high test coverage
- Include both unit and integration tests
- Mock external dependencies (LLM APIs, etc.)

## Project Structure

```
lattice/
├── lattice/           # Main package
│   ├── core/         # Enricher, config, checkpoint, exceptions
│   ├── steps/        # Step protocol + built-in steps (LLMStep, FunctionStep)
│   ├── pipeline/     # DAG resolution + column-oriented execution
│   ├── schemas/      # Pydantic models
│   ├── data/         # FieldManager (CSV field definitions)
│   └── utils/        # Logging
├── tests/            # Test suite
├── examples/         # Usage examples
└── docs/            # Documentation
```

## Testing

### Running Tests
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=lattice

# Run specific test file
python -m pytest tests/test_enricher.py

# Run with verbose output
python -m pytest -v
```

### Writing Tests
- Place test files in the `tests/` directory
- Name test files `test_*.py`
- Use descriptive test function names
- Group related tests in test classes
- Use fixtures for common test setup

## Documentation

- Update relevant documentation for any changes
- Add docstrings to new functions and classes
- Update examples if API changes
- Check that documentation builds correctly

## Getting Help

- Check existing issues and documentation
- Ask questions in pull request comments
- Create an issue for bugs or feature requests

## Code Review Process

All submissions require code review. We use GitHub pull requests for this purpose.

### Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests are included and pass
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance implications considered
- [ ] Security implications considered

Thank you for contributing to Lattice!