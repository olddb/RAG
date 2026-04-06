# Agent Collaboration Rules for RAG Project

## Git Operations Policy

⚠️ **CRITICAL: NEVER perform git operations without explicit user approval**

### Commits
- **NEVER** commit changes without asking the user first
- Always show what will be committed and ask for approval
- Wait for explicit "yes" or similar confirmation before running `git commit`

### Push
- **NEVER** push to remote without explicit user request
- ALWAYS ask before pushing, even if changes are already committed
- Wait for explicit user approval before running `git push`

### Pull / Rebase / Other operations
- Ask before running any destructive git operations
- Ask before force operations (force push, hard reset, etc.)

## Workflow

1. Make code changes as requested
2. Show the user what has changed (via git diff/status)
3. **ASK**: "Ready to commit these changes. Commit message: [message]. Should I proceed?"
4. Wait for user approval
5. Only after approval: run git commit
6. **ASK**: "Changes committed. Ready to push to GitHub?"
7. Only after push approval: run git push

## Exception
Only proceed without asking if the user explicitly says something like:
- "commit and push"
- "go ahead"
- "yes, push it"

Otherwise, **always ask first**.

---
Last Updated: 2026-03-22
Violation of these rules is a critical failure.
