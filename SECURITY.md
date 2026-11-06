# Security

## Audit, 6 November 2026

- **Method:** Cloudflare's [security-audit skill](https://github.com/cloudflare/security-audit-skill), **quick** profile.
  - 4 reconnaissance passes, 4 hunters, 1 coverage critic and 8 independent verifiers.
- **Scope:** commit `f64d78e`: `scripts/`, `tests/`, the requirements files and `.gitignore`.
- **Limits:**
  - **Source-only review.** The skill only runs target code inside an OS-enforced sandbox, and none was available on the Windows audit machine. As a result, no finding could be marked *confirmed*.
  - **Partial coverage.** The quick profile is a partial pass, not exhaustive coverage.

### Threat model

Skinnerbek Studio runs on the user's own machine and has no accounts. Its stores of personal data are reference recordings and saved voices, which are biometric data. The risks are other software, other web pages or other people reaching the local server, and inputs that exhaust the machine.

### Results

| # | Lead (needs validation, no severity) | Fix | Regression test |
|---|---|---|---|
| 1 | No login and no Host check. A web page using DNS rebinding, a LAN client (`--host`) or a public link (`--share`) could read and delete voices. | The server rejects requests whose Host is not `127.0.0.1`/`localhost`/`[::1]`. `--share` and a non-local `--host` refuse to start without `--auth USER:PASSWORD`. The bundled demo, which has no login, is limited to local use. | `test_foreign_host_is_refused_when_local_only`, `test_exposure_without_a_login_is_refused` |
| 2 | The recorder's value was trusted as a server file path, so a caller could copy files from the working or temp folder. | The value is accepted only if it is a file inside Gradio's upload folder. | `test_recorder_value_must_be_an_upload` |
| 3 | A restored session's inner `params.prefix` and `parts` were unchecked, allowing writes and reads outside `output/parts`. | Both are validated: the prefix is a plain name, and parts must sit beside the session file. Output file names are also sanitised in one place. | `test_session_with_unsafe_params_prefix_is_refused`, `test_session_parts_must_sit_beside_the_session` |
| 4 | Generation was not serial. A closed tab left its job running, and one shared stop flag was cleared by any new request. | Model use is serialised behind a lock. Each run has its own stop flag, set when the browser disconnects. Stop reaches every run. | `test_stop_reaches_every_running_generation` |
| 5 | Audio was fully decoded before any size or length check. | Files over 50 MB, longer than 600 s or with more than 8 channels are refused before decoding. Uploads are capped at 50 MB. | `test_oversized_clip_is_refused_before_decoding` |
| 6 | Unbounded script text could drive quadratic regex backtracking. | Scripts are capped at 20,000 characters. Number runs over 40 characters are split before matching, and the Markdown-link pattern no longer backtracks. | `test_long_script_is_refused`, `test_number_runs_stay_fast` |
| 7 | Any site could frame the app, and deletes take only clicks. | Every response sends `X-Frame-Options: SAMEORIGIN` and `Content-Security-Policy: frame-ancestors 'self'`. | `test_local_hosts_are_served_with_frame_protection` |

**Rejected:** raw regex rules in the lexicon. The lexicon is operator-written configuration. Even so, a rule with an invalid replacement template is now skipped instead of breaking every generation.

**Live checks (after the fixes):**
- A request with `Host: evil.example` returns `403`.
- Responses carry both framing headers.
- All 173 tests pass.

### Hardening also applied

- Gradio usage analytics are off.
- Toast messages escape names.
- The legacy voice-clip path is limited to `voices/` and `samples/`.
- The preset speaker is checked against the built-in list.
- `.gitignore` now covers `.gradio/`, `*.webm`, `*.opus`, `*.aac`, `*.pem`, `*.key` and `credentials*.json`.

### Not changed

- **Transitive dependencies** are not hash-locked. The direct ones are pinned with `==`.
- **Hugging Face model revisions** are not pinned. The models load as safetensors with no remote code.
- **Google Fonts:** the theme loads them over the network, with local fallbacks.

### Not covered by this audit

- The voice name in Speak becoming part of output file names. This was deferred by the quick profile, but fixed anyway by the shared file-name sanitiser.
- The launchers (`start.bat`, `start.command`, `scripts/start.py`), which were added after the audited commit:
  - They download uv 0.13.5 from GitHub and check its published SHA-256 before running it.
  - Python, packages and models then come from python-build-standalone, PyPI and Hugging Face over HTTPS.

## Reporting

This repository is private and proprietary. Report a vulnerability to the owner directly rather than in a public issue.
