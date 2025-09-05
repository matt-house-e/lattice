# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2024-09-05

### Added
- Complete architecture refactoring for production readiness
- Clean package structure with proper module organization
- Enhanced configuration system with development/production presets
- Comprehensive error handling with custom exceptions
- Async processing support for FastAPI integration
- Type hints throughout the codebase
- Professional packaging with pyproject.toml
- Factory methods for easy chain creation
- Progress tracking and callback support
- Retry logic for robust LLM processing

### Changed
- **BREAKING**: Refactored from flat file structure to proper package
- **BREAKING**: Simplified API - `TableEnricher` now uses composition pattern
- **BREAKING**: Unified configuration in `EnrichmentConfig` class
- Improved import system with clean public API
- Enhanced field management with better validation
- Better separation of concerns between components

### Removed
- Redundant old implementation files
- Over-engineered logging system
- Unused utility functions and dead code
- Complex try/except import blocks

### Fixed
- Import issues with standalone script execution
- Max tokens configuration for OpenAI models
- Memory usage optimization
- Error handling edge cases

## [0.1.0] - 2024-09-05

### Added
- Initial working implementation
- Basic CSV enrichment functionality
- LangChain integration for LLM processing
- Field category management system
- Vector store support (experimental)
- Simple chain implementations
- Basic configuration management
- Progress tracking with tqdm
- Example data and test scripts

### Features
- Row-by-row CSV processing
- Customizable field definitions
- OpenAI GPT integration
- Progress bars and logging
- Configurable processing delays
- Error recovery and partial results

---

## Release Notes

### v0.2.0 - Production Ready Architecture
This major release represents a complete architectural overhaul focused on production readiness, maintainability, and developer experience. The core functionality remains the same, but the codebase is now clean, well-organized, and ready for serious use.

**Migration Guide**: If upgrading from v0.1.0, update your imports:
```python
# Old way
from enrichment import TableEnricher
from field_manager import FieldManager

# New way  
from lattice import TableEnricher, FieldManager
```

### v0.1.0 - Initial Release
First working version with basic CSV enrichment capabilities. Functional but with technical debt that has been addressed in v0.2.0.