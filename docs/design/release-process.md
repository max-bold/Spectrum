# Release process

Development is committed to `dev`. Push an annotated version tag on the chosen
`dev` commit to start `.github/workflows/release.yml`, for example:

```bash
git switch dev
git push origin dev
git tag -a v0.3.2a -m "BM Spectrum v0.3.2a"
git push origin v0.3.2a
```

Before tagging, update the application version, splash artwork, README and
changelog as appropriate. A suffix such as `a` may label a packaging revision
without changing the embedded application version.

Actions verifies that the tagged commit belongs to `dev`, removes development
files from an isolated checkout, and archives the resulting Git tree. Windows,
macOS Intel and macOS Apple Silicon all test and build that exact archive.
Only after all three builds pass does the publish job replace the tracked tree
of `main` with those same sources and create a single release commit. It pushes
without force, preserving the independent history of `main`, then publishes
the GitHub Release and its three ZIP assets against the original tag on `dev`.
No release branch or additional tag on `main` is needed. GitHub's automatic
source downloads contain the original dev tree; `main` contains the clean tree.

The cleanup excludes `.vscode`, `Spectrum.code-workspace`, `TODO.md`, `info`,
`sandbox`, `artifacts`, `spectrum_app_old`, `tests_old`, `utils`, scratch root
screenshots and `docs/analog_thd_assets`. Application sources, tests, packaging,
documentation, changelog and branding assets remain.

Release runs are serialized. The workflow needs permission to write repository
contents and rules for `main` must permit its push. If testing or building fails,
`main` and GitHub Releases remain unchanged. If publication fails after the main
push, rerun the failed job; an identical tree does not create another commit.
Never move or reuse a published tag. Inspect the Actions run and all three ZIP
assets after release. Manual workflow dispatch builds clean sources without
updating `main` or publishing a release.

Historical tags through `v0.3.2` retain their original locations.
