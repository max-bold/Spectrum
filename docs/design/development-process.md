# Development process

## Working agreement

- Architectural discussion is read-only unless implementation is explicitly
  requested.
- Before an architectural change, agree on its goal, affected files, result,
  verification, and non-goals.
- Implement one architectural concept at a time in a small, runnable step.
- Stop and discuss when implementation reveals a new architectural decision or
  requires a wider scope than agreed.
- Do not mix unrelated cleanup or refactoring into an iteration.
- After an iteration, report what changed, how it was verified, deviations from
  the plan, and remaining work.
- Keep communication concise. Expand explanations only when requested or when a
  risk cannot be understood without the additional detail.

## Implementation order

For public architectural interfaces:

1. Agree on responsibilities and ownership.
2. Agree on the public API.
3. Add a minimal skeleton and contract tests.
4. Implement one end-to-end scenario.
5. Review the API before migrating more functionality.

Commits should normally contain one coherent concept. Commits and pushes are
made only as an explicit part of the requested task.
