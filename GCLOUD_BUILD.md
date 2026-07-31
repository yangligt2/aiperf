# Building the AIPerf Image with Google Cloud Build

Notes for building this repo's Docker image on Cloud Build and pushing it to
Artifact Registry, for side-by-side comparison runs with inference-perf.

- Project: `supercomputer-testing`
- Registry: `us-central1-docker.pkg.dev/supercomputer-testing/inference-perf/aiperf`
- Tag scheme: `<YYYYMMDD>-<short commit id>` (e.g. `20260730-5ad08166a`)

## Build command

Run from the repo root:

```bash
gcloud builds submit . \
  --project=supercomputer-testing \
  --config=cloudbuild.yaml \
  --substitutions=_TAG="$(date +%Y%m%d)-$(git rev-parse --short HEAD)" \
  --async
```

`cloudbuild.yaml` in this directory builds the Dockerfile's final stage
(`runtime`, NVIDIA distroless Python) and pushes `${_IMAGE}:${_TAG}`.

Check status and resolve the pushed tag:

```bash
gcloud builds list --project=supercomputer-testing --limit=3
gcloud artifacts docker tags list \
  us-central1-docker.pkg.dev/supercomputer-testing/inference-perf/aiperf \
  --project=supercomputer-testing
```

## Non-obvious requirements (each one caused a failed build)

1. **BuildKit is mandatory.** The Dockerfile uses `COPY --chmod`, which the
   legacy Docker builder rejects (`the --chmod option requires BuildKit`).
   Cloud Build's `gcr.io/cloud-builders/docker` defaults to the legacy
   engine, so the build step sets `DOCKER_BUILDKIT=1`. A side benefit:
   BuildKit skips stages not reachable from the final stage (`local-dev`,
   `test`, the artifact stages).

2. **Use `logging: CLOUD_LOGGING_ONLY`.** With default logging, build logs go
   to the project's default GCS logs bucket, which sits outside the VPC-SC
   perimeter and is not readable by normal user accounts here - a failed
   build cannot be diagnosed (`gcloud builds log` returns AccessDenied).
   With Cloud Logging, logs are readable via:

   ```bash
   gcloud logging read \
     'resource.type="build" resource.labels.build_id="<BUILD_ID>"' \
     --project=supercomputer-testing --format="value(textPayload)" --order=desc
   ```

   This also means the plain `gcloud builds submit --tag <image> .` shortcut
   is not usable in this project: it offers no way to set the logging option.

3. **Machine type `E2_HIGHCPU_32`.** The `env-builder` stage compiles ffmpeg
   from source with `make -j$(nproc)`; on the default 2-vCPU machine that
   stage dominates the build. On 32 vCPUs the whole build takes ~4.5 min.
   The default 10 min `timeout` would also be exceeded on small machines;
   the config sets 3600 s.

## Details verified against the repo

- The final Dockerfile stage is `runtime`; a plain `docker build .` (no
  `--target`) builds it, so no target selection is needed in the config.
- The package version is static in `pyproject.toml` (hatchling). The source
  upload excludes `.git` (ignore list derived from `.gitignore`), which is
  safe because nothing in the build reads git metadata.
- No `--mount=type=cache` or `# syntax=` directives are used, so no
  BuildKit frontend pinning is needed - the `DOCKER_BUILDKIT=1` env var is
  sufficient.
