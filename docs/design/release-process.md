# Release process

Development is committed to `dev`. Releases are prepared in a separate
`release/vX.Y.Z` branch and transferred to `main` as one release commit.
The version tag is placed on that final `main` commit.

1. Update `spectrum_app/version.py`, the splash version artwork, README,
   changelog and TODO. Run the test suite and verify a local packaged build.
2. Commit and push `dev`.
3. Create `release/vX.Y.Z` from that development commit in a separate checkout.
4. Remove development-only files from the release checkout: `.vscode`,
   `Spectrum.code-workspace`, `TODO.md`, `info`, `sandbox`, `artifacts`,
   `spectrum_app_old`, `tests_old`, `utils`, scratch screenshots and experimental
   `docs/analog_thd_assets`. Keep application sources, tests, packaging,
   documentation, changelog and branding assets.
5. Check README links and dependencies, run tests in the cleaned checkout,
   then commit and push the release branch.
6. Start another checkout at the latest `origin/main`. Replace its tracked
   contents with the prepared release tree and commit `Release vX.Y.Z`.
   This preserves the independent history of `main`; do not merge all of `dev`.
7. Verify that the release branch and new `main` commit have identical trees,
   run tests, and push `main` without force.
8. Create an annotated `vX.Y.Z` tag on the final `main` commit and push it.
   Never reuse or move an already published version tag.
9. Follow `.github/workflows/release.yml`: all three builds (Windows x86_64,
   macOS Intel, macOS Apple Silicon) must pass before the publish job runs.
   Verify the GitHub Release and its three ZIP assets.

Keep the development checkout on `dev`; cleanup applies only to the separate
release checkout. If CI fails, diagnose it before deciding how to correct the
release; do not silently replace published tags or binaries.

Historical tags differ: `v0.3.1` points to the prepared release-branch commit,
while `v0.3` points to the final `main` commit. Their corresponding trees match.
Starting with `v0.3.2`, tags consistently identify the final `main` commit.
