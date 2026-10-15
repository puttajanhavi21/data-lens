# Verification notes

Verified on 14 October 2026 using Windows and Python 3.12.14.

## Automated checks

Command: `python -m unittest discover -s tests -v`

Result: **33 tests passed**.

The checks cover:

- Valid and malformed CSVs, upload limits, descriptive statistics, missing values, and duplicate handling.
- Cleaning without changing the source data, reset, CSV download, and formula-leading text in exports.
- Flask upload, sample loading, rendered results, all three model workflows, and saved run history.
- HTML escaping, form-token checks, and separation of browser-session datasets.
- Classification, regression, clustering, baseline comparisons, and invalid model inputs.
- Fitting preprocessing on training rows only and keeping exploration cleaning out of training.
- Local explanations, required AI consent, summary-only AI payloads, and provider-error fallback.

The three bundled sample workflows complete using synthetic data. Their metrics are learning examples, not evidence of real-world business performance.

## Verification limits

AI success and failure paths were tested with simulated responses. No live paid OpenAI request was made. Configure your own API access to verify live generative explanations.

The Flask server started successfully and the templates were exercised by integration tests. An interactive browser preview could not be attached in the build environment, so visual layout and browser-specific behavior were not independently verified.

Python 3.11, macOS, and Linux were not separately tested. The project uses portable Python paths and includes setup instructions for those operating systems.

The ZIP excludes runtime uploads, databases, session secrets, virtual environments, and API keys. The application creates its local runtime files when you run it.
