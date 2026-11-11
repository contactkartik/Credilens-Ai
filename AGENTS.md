# CrediLens AI - Project Rules

## Core Principles

1. **Never hardcode ML predictions** - All predictions must come from trained models
2. **Never fabricate accuracy, precision, recall, F1 or other results** - Use actual experimental results only
3. **Never use an LLM API as a replacement for our trained ML model** - LLMs may be used for辅助 tasks but not as the primary credibility detection mechanism
4. **Keep ML, backend and frontend code separate** - Maintain clear separation of concerns
5. **Do not modify unrelated files when implementing a feature** - Only modify files relevant to the current task
6. **Before making major changes, inspect the relevant existing files** - Understand the codebase before modifying
7. **Use actual experimental results in the final project** - No fake or placeholder results in the final submission
8. **Keep the code modular and easy to explain during a college viva** - Write clean, well-documented code

## Development Guidelines

- Follow the existing code structure and conventions
- Add appropriate tests for new features
- Document complex logic for viva preparation
- Use version control with meaningful commit messages
- Review all changes before committing

## ML Pipeline

- Use the `ml/` directory for all ML-related code
- Store raw data in `data/raw/`
- Store processed data in `data/processed/`
- Save trained models in `models/`
- Document experiments in `notebooks/`

## Backend

- Use FastAPI for the backend API
- Follow the structure in `backend/app/`
- Use environment variables for configuration
- Implement proper error handling

## Frontend

- Use React for the frontend
- Keep UI components modular
- Follow React best practices
- Ensure responsive design
