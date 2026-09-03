# Deploy to Azure

**Target: Azure Container Apps** — the serverless container host. Scales to zero (≈ $0
idle), HTTPS + hostname included, secrets are first-class, no VM or App Service plan.

### Why not Azure Functions

The agent keeps conversation memory **in-process** (LangGraph `MemorySaver`, keyed by
`session_id`). Functions gives no instance affinity, so memory would vanish between
requests. Container Apps runs a normal long-lived process; with `--min-replicas 1` the
memory persists, with `0` it resets only when the app scales to zero (fine for a demo).
The heavy `langgraph` / `langchain` dependency tree and the static-file mount also fit a
container far better than a Functions package.

---

## One-time setup

### 1. Pick a subscription

```bash
az login
az account set --subscription "<SUBSCRIPTION_ID>"
```

### 2. Create a federated (OIDC) identity for GitHub Actions

No client secret is stored — GitHub exchanges a short-lived token.

```bash
APP_ID=$(az ad app create --display-name "voice-agent-gha" --query appId -o tsv)
az ad sp create --id "$APP_ID"

SUB=$(az account show --query id -o tsv)
az role assignment create --assignee "$APP_ID" --role Contributor \
  --scope "/subscriptions/$SUB"

# Trust pushes to main and manual runs from this repo
az ad app federated-credential create --id "$APP_ID" --parameters '{
  "name": "gha-main",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:<OWNER>/<REPO>:ref:refs/heads/main",
  "audiences": ["api://AzureADTokenExchange"]
}'

echo "AZURE_CLIENT_ID=$APP_ID"
az account show --query '{AZURE_TENANT_ID:tenantId, AZURE_SUBSCRIPTION_ID:id}' -o table
```

### 3. Add repo variables + secret

GitHub → repo → **Settings → Secrets and variables → Actions**

| Kind      | Name                     | Value                              |
| --------- | ------------------------ | ---------------------------------- |
| Variable  | `AZURE_CLIENT_ID`        | `$APP_ID` from step 2              |
| Variable  | `AZURE_TENANT_ID`        | tenant id from step 2              |
| Variable  | `AZURE_SUBSCRIPTION_ID`  | subscription id from step 2        |
| Secret    | `ASSEMBLYAI_API_KEY`     | your key (STT **and** LLM Gateway) |

### 4. Deploy

Push to `main`, or run **Actions → Deploy to Azure Container Apps → Run workflow**.
The job builds the image in the cloud (ACR Tasks — no Docker in CI), creates the
resource group / Container Apps environment / registry on first run, deploys a
revision, wires the API key as a secret, and prints the URL to the run summary.

---

## Knobs

Edit `env:` in [.github/workflows/deploy-azure.yml](.github/workflows/deploy-azure.yml):

| Setting          | Default              | Notes                                                         |
| ---------------- | -------------------- | ------------------------------------------------------------ |
| `LOCATION`       | `eastus`             | Only affects the two small API calls; STT is edge-routed.    |
| `APP_NAME`       | `assemblyai-voice-agent` | becomes the hostname prefix                             |
| `--min-replicas` | `0`                  | set to `1` to keep LangGraph memory warm (≈ a few $/mo)      |
| `--max-replicas` | `2`                  | each replica has its own `MemorySaver` — see below           |

**Multi-instance memory:** `MemorySaver` is per-process. With `max-replicas > 1` a
follow-up turn can land on a replica that never saw the earlier turns. For a shared,
durable store add `langgraph-checkpoint-postgres` (or `-redis`) and swap the
checkpointer in `app/agent.py`; or keep `max-replicas 1`.

---

## Test the container locally

```bash
docker build -t voice-agent .
docker run --rm -p 8000:8000 --env-file .env voice-agent
# http://localhost:8000
```

## Tear down

```bash
az group delete -n voice-agent-rg --yes --no-wait
```
