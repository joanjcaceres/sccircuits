## General GitHub PR Workflow

- For every repository and project, request GitHub Copilot as a reviewer on pull requests by default.
- Use the supported GitHub CLI reviewer value `@copilot` when available: `gh pr edit <PR> --add-reviewer "@copilot"`.
- If the installed `gh` version does not support `@copilot`, use a current official GitHub CLI binary rather than the older system `gh`.

## GitHub Issue Workflow

- When a project-level decision, architectural tradeoff, environment/tooling question, or deferred follow-up is identified, create or update a GitHub issue to track the reasoning and next action instead of relying only on chat context.
- Prefer one focused issue per decision or follow-up, with context, questions to answer, acceptance criteria, and links to related PRs when relevant.
- Use issues as the default planning memory across repositories before changing direction or making a broad refactor.

## Markdown Math Rendering

- When editing Markdown documentation with LaTeX, write display math in robust fenced form or simple `$$` blocks that do not contain Markdown-sensitive line starts.
- Avoid starting a line inside `$$` math with Markdown control characters such as `-`, `=`, `*`, or `#`; GitHub can interpret those as Markdown instead of math.
- Prefer `aligned`, `bmatrix`, or single-line equations for multiline derivations. If a continuation line needs a minus sign, use a safe prefix such as `{}-` or keep the expression on one line.
- Avoid long comma-separated lists of inline math expressions when GitHub preview has rendered similar patterns poorly. Use a display equation, table, or plain-code identifiers instead.
- After changing math-heavy Markdown, inspect the rendered GitHub preview or run the docs build and check likely problem sections before considering the documentation finished.
