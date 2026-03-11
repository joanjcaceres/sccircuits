# Contributing to SCCircuits

Thank you for considering contributing to SCCircuits! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository and clone your fork:
```bash
git clone https://github.com/yourusername/sccircuits.git
cd sccircuits
```

2. Install Pixi (if needed):
```bash
# macOS (Homebrew)
brew install pixi

# Cross-platform installer (alternative)
# curl -fsSL https://pixi.sh/install.sh | sh
```

3. Install the project in editable mode inside the Pixi environment:
```bash
pixi run -e sccircuits install-dev
```

This editable install uses `pip --no-deps` to avoid replacing Pixi-managed binary dependencies.
Use a dedicated Pixi environment for this repository instead of a shared global Conda environment.

## Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **flake8** for linting  
- **mypy** for type checking

Run formatting and checks:
```bash
pixi run -e sccircuits format
pixi run -e sccircuits lint
pixi run -e sccircuits typecheck
```

## Running Tests

Run tests with pytest:
```bash
pixi run -e sccircuits test
```

For coverage report:
```bash
pixi run -e sccircuits coverage
```

## Dependency Updates

When updating dependencies, use:

```bash
pixi update
pixi run -e sccircuits deps-check
pixi run -e sccircuits test
```

If conflicts appear after upgrades, reset the Pixi environment and reinstall:

```bash
pixi clean
pixi run -e sccircuits install-dev
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
- Whether you used Pixi default environment or a custom environment
- Minimal code example to reproduce the issue
- Full error traceback

## Questions

For questions about development, please open an issue with the "question" label.

Thank you for contributing!
