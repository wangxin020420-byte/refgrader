# SCALE Pe and graph pilot

Independent reconstruction of the extraction and graph portion only. The user
selected glm-5.3 in place of Llama-3.1-8B-Instruct. Account/model availability
must be verified by the actual API response; local preflight does not call API.
Uses the existing adapted 885 training records only. Does not open validation
or test files, train a scorer, generate synthetic answers, or replace old runs.

Run stage1.py prepare --root ROOT; extract_api.py --root ROOT for local
preflight; add --execute to send six sequential requests. Default endpoint is
the BigModel general API, configurable using --base-url. No API key is stored.
Responses are saved for reuse. Interrupted requests with uncertain completion
are not automatically retried. Preserve the run and inspect errors before retry.

Schema validation verifies source spans, not semantic correctness. Review
pilot_extraction_review.txt before scaling to the entire training split.
The Pe prompt and type examples are reconstructed, not author supplied.
Graph modes prose and equation4 expose the paper's soft-edge ambiguity and
produce separate artifacts; neither is silently selected as the final protocol.
Typed nodes and directed edges are implementation choices recorded explicitly.
