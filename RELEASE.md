# Release checklist

Follow the steps in order and tick each box. Every step says what to run and what to check.
If a check fails, stop and go to "When something goes wrong" at the end.

Set the two versions once, in the terminal you will use for the whole release:

```bash
VERSION=X.Y.Z   # the new version, without the leading v
PREV=A.B.C      # the version prod runs now
```

Keep at hand: the test and prod API keys, and an `employee.json` file holding the example body
of the API section of the README.

## 1. Before you start

- [ ] The pull requests to ship are merged, or ready to merge, and the CI of `main` is green
  (Actions tab).
- [ ] `VERSION` is new. A version already published on GHCR is never reused, even after a
  failed release: use the next patch number instead.
- [ ] Settings > Environments > `prod` shows "Required reviewers" with `dapitch666`. Without it,
  prod deploys with no approval.
- [ ] Both environments answer with the previous version:

  ```bash
  curl -s https://attrition-test.laprade.eu/health   # from the local network only
  curl -s https://attrition.laprade.eu/health
  ```

  Expected: `"status":"ok"`, `"version":"$PREV"`, and the right `environment`.

## 2. Release commit

Make it the last commit of the branch to release.

- [ ] Set the version in `pyproject.toml` and `uv.lock`:

  ```bash
  uv version "$VERSION"
  ```

- [ ] In `docker-compose.yml`, set the tag of the `x-image` line to `$VERSION`.
- [ ] In the Quick start of `README.md`, set the version of the download command to `v$VERSION`.
- [ ] Check that the version is everywhere:

  ```bash
  grep -n "$VERSION" pyproject.toml docker-compose.yml README.md
  git diff --stat
  ```

  Expected: one line in `pyproject.toml`, one in `docker-compose.yml`, at least two in
  `README.md`, and changes in those three files plus `uv.lock`.

- [ ] Commit and push:

  ```bash
  git commit -am "chore: release $VERSION"
  git push
  ```

- [ ] The CI of the branch is green. The `lint` job fails when `docker-compose.yml` and
  `pyproject.toml` disagree.

## 3. Merge

- [ ] Open the pull request, wait for the green checks, and merge it with a merge commit.
- [ ] Update the local `main`:

  ```bash
  git switch main && git pull
  git log --oneline -1
  ```

  Expected: the merge commit of the pull request.

## 4. Tag

- [ ] Tag the merge commit with a one-line summary of the release, and push the tag:

  ```bash
  git tag -a "v$VERSION" -m "<summary of the release>"
  git push origin "v$VERSION"
  ```

## 5. Publication and test deployment

In the Actions tab, open the "Publish" run of `v$VERSION`.

- [ ] The three CI jobs are green.
- [ ] "Build and publish the image" is green. The image appears under Packages on the
  repository page with the tag `$VERSION`.
- [ ] "Deploy to test" is green.
- [ ] "Deploy to prod" shows "Waiting for review". Do not approve yet.

## 6. Check test

Portainer redeploys in the background: allow one or two minutes.

- [ ] In Portainer, stack `attrition-test`: `init` has exited with code 0, `api` is running, and
  its image is `...:$VERSION`.
- [ ] Health, from the local network:

  ```bash
  curl -s https://attrition-test.laprade.eu/health
  ```

  Expected: `"version":"$VERSION"`, `"environment":"test"`.

- [ ] A prediction with the test key:

  ```bash
  curl -s -X POST https://attrition-test.laprade.eu/predict \
    -H "X-API-Key: <test key>" -H "Content-Type: application/json" -d @employee.json
  ```

  Expected: a JSON with `score`, `alert` and `model_version`.

- [ ] The same call without the key answers 401.
- [ ] Anything the release changed works on test.

## 7. Prod

- [ ] In the run, click "Review deployments", tick `prod`, and approve.
- [ ] "Deploy to prod" is green. Its smoke test step prints `Prod runs version $VERSION.`
- [ ] A prediction with the prod key:

  ```bash
  curl -s -X POST https://attrition.laprade.eu/predict \
    -H "X-API-Key: <prod key>" -H "Content-Type: application/json" -d @employee.json
  ```

- [ ] Delete the merged branch on GitHub and locally (`git branch -d <branch>`).

The release is done.

## When something goes wrong

Prod is only touched in step 7. Until you approve, a failure leaves prod on `$PREV`.

### The CI of the release branch fails

Read the failing job. "docker-compose.yml pins [...]" means the tag of the `x-image` line does
not match `pyproject.toml`. Fix the file, commit, push.

### "Publish" fails before the image is published

The tag check, the check that the commit is on `main`, or the CI failed. Nothing was published
or deployed. Delete the tag, fix the cause, and tag again with the same version:

```bash
git tag -d "v$VERSION"
git push origin ":refs/tags/v$VERSION"
```

Check under Packages that `$VERSION` does not exist before tagging again.

### The image is published but the release is bad

Do not reuse the version. Reject the pending prod deployment if there is one, fix the code, and
release the next patch version from step 2.

### "Deploy to test" fails

The webhook call failed: the job prints the error. A 401 or 404 usually means a wrong
`PORTAINER_WEBHOOK_URL` or `HOOK_TOKEN` in the `test` environment. A 502 or no answer means the
webhook proxy or Portainer is down on the NAS. Fix the cause, then click "Re-run failed jobs".

### Test does not come up, or does not work

Reject the prod deployment: prod stays on `$PREV`. Read the logs of `init` and `api` in
Portainer. Then roll test back (see below, with the stack `attrition-test`), or fix the code and
release the next patch version.

Warning: `main` now pins `$VERSION`. Any redeployment of a stack that follows `main`, from
Portainer or from a webhook, installs `$VERSION`.

### The smoke test fails

Read the last response printed by the job.

- `"version"` is still `$PREV`: the new container is not up yet, often a slow image pull. Wait
  until Portainer shows `api` running on `$VERSION`, then click "Re-run failed jobs".
- No answer, or a 502: `api` is not running. Read the logs of `init` and `api` in Portainer.
- `"environment"` is not `prod`: check `APP_ENV` in the variables of the `attrition-prod` stack.

If prod does not recover within a few minutes, roll it back.

### Roll back a stack

Quick route, in Portainer:

1. Open the stack (`attrition-prod` or `attrition-test`) and its "Edit stack settings" window.
2. In "Repository reference", pick `refs/tags/v$PREV` instead of `refs/heads/main`. Leave the
   other settings and the webhook as they are.
3. Tick "Redeploy" and click "Save settings".
4. Check `/health`: it must report `$PREV`.

The database volume is kept. `init` only adds missing tables and resets the rights, so the
previous version runs on the same database, unless the release changed an existing table. In
that case, check the change before rolling back.

The stack no longer follows `main`. Once a fixed version is published, set the reference back to
`refs/heads/main` before you approve its prod deployment, otherwise the webhook redeploys
`$PREV` and the smoke test fails.

Fallback, when the Portainer interface cannot be reached but its webhooks can: open a pull
request on `main` that reverts the release commit (`git revert <sha of "chore: release $VERSION">`)
and merge it. `main` then pins `$PREV` again. Call the webhook of the stack. GitHub secrets cannot be read
back, so keep both webhook URLs (with the host of the proxy) and the token in the password
manager:

```bash
curl --fail-with-body -X POST -H "X-Hook-Token: <token>" "<webhook URL>"
```

The next release uses the next patch version.
