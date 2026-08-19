# AI-Compliance-Investigation-Agent
Production-ready AI compliance investigation agent featuring RAG, structured outputs, tool calling, safety guardrails, evaluations, and end-to-end observability.

## Local backend setup

```bash
pyenv local 3.12.10
python -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://pypi.org/simple -e '.[dev]'
uvicorn compliance_agent.main:app --reload
```

Configure the default Groq provider in the ignored `.env.local`:

```dotenv
AI_PROVIDER=groq
GROQ_API_KEY=replace-with-a-local-key
GROQ_MODEL=openai/gpt-oss-20b
```

Never paste a real key into source code, documentation, Git, Postman, or chat. OpenAI remains an
optional provider selected with `AI_PROVIDER=openai` and its corresponding local configuration.

The local API is available at `http://127.0.0.1:8000`:

- `GET /health` checks whether the process is alive.
- `GET /ready` checks whether required configuration is present without exposing secrets.
- `GET /api/v1` is the versioned API entrypoint.
- `POST /api/v1/cases/validate` validates a compliance case without storing it or calling AI.
- `POST /api/v1/alerts/evaluate/transactions` evaluates fictional portfolio rules without AI.
- `POST /api/v1/recommendations` returns a schema-validated AI recommendation for human review.
- `POST /api/v1/tools/execute` demonstrates bounded read-only tool execution without AI.
- `POST /api/v1/investigations/run` runs planning, local tools, and a final recommendation.
- `GET /docs` opens the generated OpenAPI interface.

### Milestone 1 curl examples

Check process liveness:

```bash
curl --request GET \
  --url http://127.0.0.1:8000/health \
  --header 'Accept: application/json'
```

Check whether required configuration is present:

```bash
curl --request GET \
  --url http://127.0.0.1:8000/ready \
  --header 'Accept: application/json' \
  --header 'X-Request-ID: postman-ready-check'
```

Check the versioned API entrypoint:

```bash
curl --request GET \
  --url http://127.0.0.1:8000/api/v1 \
  --header 'Accept: application/json'
```

Import `postman/AI-Compliance-Investigation-Agent.postman_collection.json` into Postman for the
same requests and basic response tests. Its `baseUrl` collection variable defaults to
`http://127.0.0.1:8000`.

### Milestone 2 curl examples

Validate the realistic sample case:

```bash
curl --request POST \
  --url http://127.0.0.1:8000/api/v1/cases/validate \
  --header 'Accept: application/json' \
  --header 'Content-Type: application/json' \
  --data @examples/sample_case.json
```

Demonstrate a safe validation failure:

```bash
curl --request POST \
  --url http://127.0.0.1:8000/api/v1/cases/validate \
  --header 'Accept: application/json' \
  --header 'Content-Type: application/json' \
  --header 'X-Request-ID: invalid-case-demo' \
  --data '{"case_id":"INCOMPLETE-CASE"}'
```

The first call returns a deterministic summary. The second returns HTTP 422 using the same stable
error envelope as other API failures. Neither call stores data or spends model tokens.

### Milestone 3 curl examples

Evaluate a fictional transaction batch that triggers two alerts:

```bash
curl --request POST \
  --url http://127.0.0.1:8000/api/v1/alerts/evaluate/transactions \
  --header 'Accept: application/json' \
  --header 'Content-Type: application/json' \
  --data @examples/sample_transaction_activity.json
```

The sample deliberately triggers fictional thresholds: five qualifying non-family transfers and
AED 25,000 in monthly outbound volume. The response contains typed, explainable mock alerts; it
does not call AI, store a case, contact a screening provider, or actually block an account.

### Milestone 4 curl example

Generate a structured recommendation from the validated sample case:

```bash
curl --request POST \
  --url http://127.0.0.1:8000/api/v1/recommendations \
  --header 'Accept: application/json' \
  --header 'Content-Type: application/json' \
  --header 'X-Request-ID: recommendation-demo' \
  --data @examples/sample_case.json
```

This endpoint makes a provider API call, using Groq by default. It uses Structured Outputs,
disables provider-side response storage for the request, returns safe model/latency/token
metadata, and requires human review. It rejects recommendations that cite unknown case IDs or
claim unsupported tool/policy evidence.

### Milestone 5 curl examples

Run the complete bounded investigation workflow:

```bash
curl --request POST \
  --url http://127.0.0.1:8000/api/v1/investigations/run \
  --header 'Accept: application/json' \
  --header 'Content-Type: application/json' \
  --header 'X-Request-ID: bounded-investigation-demo' \
  --data @examples/sample_case.json
```

The planning model can request only `lookup_customer_risk` and `summarize_transactions`. The
application validates arguments, enforces the tool-call limit, prevents duplicate execution, and
runs the functions locally. A second model call receives the safe results and returns the final
structured recommendation. The workflow is read-only and never executes account controls.

Run local verification with:

```bash
python -m pytest
ruff check .
ruff format --check .
```

Copy safe configuration names from `.env.example` into the ignored `.env.local`. Never commit
real credentials.
