# Release Process

This document describes how to create releases for Valence.

## Versioning Strategy

Valence uses [Semantic Versioning](https://semver.org/) with the following conventions:

### Version Format: `MAJOR.MINOR.PATCH[-PRERELEASE]`

| Component | When to Increment |
|-----------|-------------------|
| **MAJOR** | Breaking changes to API, data formats, or core protocols |
| **MINOR** | New features, components, or backward-compatible enhancements |
| **PATCH** | Bug fixes, documentation updates, security patches |
| **PRERELEASE** | Development stages: `-alpha`, `-beta`, `-rc.N` |

### Pre-release Stages

- **`alpha`** — Feature incomplete, API unstable, for early adopters only
- **`beta`** — Feature complete, API stabilizing, seeking broader testing  
- **`rc.N`** — Release candidate, production-ready unless blockers found

### Current Stage: Alpha

Valence is in **alpha**. This means:
- Specifications are complete and stable
- Implementation is functional but incomplete
- APIs may change between minor versions
- Not recommended for production use (yet)

## Release Checklist

### Before Release

1. **Update CHANGELOG.md**
   - Move items from `[Unreleased]` to new version section
   - Add release date
   - Verify all significant changes are documented

2. **Update version in pyproject.toml**
   ```toml
   version = "X.Y.Z"  # or "X.Y.Z-alpha" for pre-releases
   ```

3. **Run tests**
   ```bash
   pytest tests/
   ruff check src/
   mypy src/
   ```

4. **Update documentation**
   - Ensure README reflects current state
   - Update STATUS.md if needed
   - Review spec/ for any needed updates

### Creating the Release

1. **Commit version changes**
   ```bash
   git add CHANGELOG.md pyproject.toml
   git commit -m "Release vX.Y.Z"
   ```

2. **Create and push tag**
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin main --tags
   ```

3. **Create GitHub release**
   ```bash
   gh release create vX.Y.Z \
     --title "Valence vX.Y.Z" \
     --notes-file RELEASE_NOTES.md
   ```
   
   Or use the GitHub web interface to create the release from the tag.

### After Release

1. **Verify release**
   - Check GitHub releases page
   - Verify tag is visible
   - Test installation from release if applicable

2. **Announce** (when appropriate)
   - Update project status
   - Notify interested parties

## Release Notes Template

```markdown
## Valence vX.Y.Z

Brief description of this release.

### Highlights
- Key feature or change 1
- Key feature or change 2

### What's New
- Detailed list from CHANGELOG

### Breaking Changes
- List any breaking changes (if any)

### Upgrading
- Migration steps if needed

### Contributors
- @username for specific contribution
```

## Spec vs Implementation Versioning

Valence has two distinct artifacts:

| Artifact | Versioning | Notes |
|----------|------------|-------|
| **Specifications** | Spec version in each doc | Design documents, may update independently |
| **Implementation** | Package version | Python code, follows semver strictly |

The package version tracks the implementation. Spec documents include their own
version headers for tracking design evolution.

When specs change significantly:
1. Update the spec document version header
2. Note the spec change in CHANGELOG under relevant implementation version
3. Implementation catches up in subsequent releases

## Hotfix Process

For urgent fixes:

1. Branch from the release tag: `git checkout -b hotfix/X.Y.Z vX.Y.Z`
2. Apply fix, test
3. Update CHANGELOG, bump PATCH version
4. Merge to main, tag new release
5. Cherry-pick to any active development branches if needed

---

*Questions? Open an issue or discussion on GitHub.*
