---
description: Run checks, commit the current step with a Conventional Commit message, and push to GitHub
---

1. Run `ruff check . && ruff format . && pytest -q`. If anything fails, stop and fix it first.
2. Run `git status` and confirm no data files, secrets, or virtualenv folders are staged.
3. Stage only the files that belong to this step with `git add <paths>`.
4. Commit with a Conventional Commit message (feat, fix, test, docs, chore, refactor).
5. Run `git push origin main`. If rejected, run `git pull --rebase origin main`, re-run the tests, and push again. Never force-push.
6. Report the commit hash, files changed, and tests run.
