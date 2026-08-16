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

The local API is available at `http://127.0.0.1:8000`:

- `GET /health` checks whether the process is alive.
- `GET /ready` checks whether required configuration is present without exposing secrets.
- `GET /api/v1` is the versioned API entrypoint.
- `POST /api/v1/cases/validate` validates a compliance case without storing it or calling AI.
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

Run local verification with:

```bash
python -m pytest
ruff check .
ruff format --check .
```

Copy safe configuration names from `.env.example` into the ignored `.env.local`. Never commit
real credentials.
