# DocuMind — Security Overview

Security is addressed across three pillars: encryption, access control, and
vulnerability management.

## 1. Encryption

- **In transit:** in production, all external traffic is served over TLS/HTTPS
  via the Kubernetes Ingress. Internal service-to-service traffic runs on the
  cluster's private network. Langfuse and model-provider APIs are called over HTTPS.
- **At rest:** Kubernetes Secrets are encrypted at rest by the cluster. The vector
  database volume can be backed by encrypted storage in a managed environment.

## 2. Access control

- API-key authentication (X-API-Key) on the sensitive endpoints (/chat, /ingest,
  /reindex); the /health endpoint is intentionally public and carries no secret.
- Secrets are injected at runtime — via env-file in Docker Compose and a Kubernetes
  Secret in the cluster (sourced from .env, never committed). They are never baked
  into images or written into manifests.
- Containers run as a non-root user, limiting blast radius if compromised.
- Production hardening: Kubernetes RBAC to scope access, and a backend-for-frontend
  so the API key never reaches the browser (the user authenticates; a server proxy
  holds the key). The current client-side key is a deliberate dev-only simplification.

## 3. Vulnerability management

- Trivy scans both container images on every push (CI). Reports CRITICAL/HIGH
  findings; in production the pipeline gates (fails) on fixable findings.
- Slim/Alpine base images and multi-stage builds minimize the attack surface
  (the frontend image ships only static files — no Node, no source).
- Dependencies are declared in requirements.txt; .dockerignore keeps .env, .git,
  and build junk out of images.

## Known limitations (deliberate, for the demo)

- Client-side API key in the frontend (see access control above for the prod fix).
- Vector store uses ephemeral storage in the demo K8s manifests; production would
  use a persistent volume or managed object storage.
