# Security notes

## Known limitations (deliberate, for the demo)

- The frontend currently sends a static API key (`X-API-Key`) from browser code.
  This is acceptable for local development only, since the key is visible in the
  browser network tab.
- Production fix: a backend-for-frontend. The browser authenticates the _user_
  (session cookie or JWT via an identity provider); a server-side proxy holds the
  real API key and forwards authorized requests. The secret never reaches the browser.

## Implemented controls

- API-key auth on /chat, /ingest, /reindex (access control).
- /health left public so it carries no secret and can back K8s probes.
- Secrets kept out of code and git (.env, .gitignore); will move to
  Kubernetes Secrets in the deployment phase.
