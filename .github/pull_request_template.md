# Pull Request Title

## Description

State the reason for your Pull Request and explain
any changes you have made.

> **Branch target.** All regular PRs must target `develop`. PRs into `main` are
> only accepted from `develop` and are exclusively used for releases — they will
> be blocked by CI otherwise. See the
> [release pipeline](https://www.notion.so/instadeep/Release-pipeline-336ed6e1dfc480288045ef94266cab48).

## Checklist for adding a new benchmark

- [ ] Benchmark class is fully implemented, including
all abstract method implementations.
- [ ] Benchmark is fully tested following our standard
testing pattern.
- [ ] Input data uploaded to HF is validated and correct.
- [ ] The corresponding documentation is added and up to date.
- [ ] The UI code is added and up-to-date and has been tested.
- [ ] Our license has been added to any Python file.

## Checklist for a release PR (`develop` → `main`)

- [ ] PR base is `main` and head is `develop`
- [ ] Version in `pyproject.toml` has been bumped
- [ ] `CHANGELOG.md` has been updated
