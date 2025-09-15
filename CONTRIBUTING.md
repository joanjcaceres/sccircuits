# Contributing to SCCircuits

Thank you for considering contributing to SCCircuits! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository and clone your fork:
```bash
git clone https://github.com/yourusername/sccircuits.git
cd sccircuits
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode with optional dependencies:
```bash
pip install -e ".[dev,interactive]"
```

## Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **flake8** for linting  
- **mypy** for type checking

Run formatting and checks:
```bash
black sccircuits/
flake8 sccircuits/
mypy sccircuits/
```

## Running Tests

Run tests with pytest:
```bash
pytest tests/ -v
```

For coverage report:
```bash
pytest tests/ --cov=sccircuits --cov-report=html
```

## Documentation

- All public functions and classes should have clear docstrings
- Use NumPy-style docstrings
- Include examples in docstrings when helpful
- Update README.md if adding new features

## Submitting Changes

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and add tests

3. Run the test suite and ensure all tests pass

4. Commit your changes with clear commit messages

5. Push to your fork and submit a pull request

## Pull Request Guidelines

- Include a clear description of the changes
- Reference any related issues
- Add tests for new functionality
- Update documentation as needed
- Ensure all CI checks pass

## Reporting Issues

When reporting bugs, please include:
- Python version
- SCCircuits version
- Operating system
- Minimal code example to reproduce the issue
- Full error traceback

## Questions

For questions about development, please open an issue with the "question" label.

Thank you for contributing!