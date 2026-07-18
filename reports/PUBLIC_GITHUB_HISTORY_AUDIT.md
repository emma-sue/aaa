# Public GitHub history audit — Codex

Audit time: 2026-07-18 14:47 UTC  
Scope: every blob reachable from every local Git ref in the public mirror; no
`.env` file was opened and matched secret content was never printed.

## Result

- 9 commits and 292 unique blobs were inspected offline (38,890,919 bytes).
- Secret-pattern findings: **0** (GitHub/OpenAI/AWS token forms, private-key
  headers and high-risk credential assignments).
- Credential-like historical paths: **0**.
- Largest ordinary-Git blob: 11,917,425 bytes; no file approaches the 95 MiB
  fail-closed limit.
- Current HEAD already excludes dataset bytes, checkpoint binaries, `.env`,
  credentials and the third-party source files listed below.

## Historical licensing residue

The following files were committed in earlier public revisions and remain
reachable even though later HEAD revisions delete them:

- `vendor/autosota/SKILL.md`;
- `vendor/researchstudio/idea_spark_SKILL.md`;
- `vendor/r2r/utils/image_utils.py` and `vendor/r2r/utils/schedulers.py`;
- five R2R data-list files under `vendor/r2r/data_dir/`.

Only the repository-level PromptIR-derived `LICENSE.md` occurs in that history;
there is no co-located upstream license/notice closure for the files above.
Therefore deletion at HEAD is not a complete redistribution remediation.

## Decision

**A controlled history squash/rewrite is required before the licensing cleanup
can be called complete.** It is not authorized to run automatically while the
backup daemon is publishing or while recovery tags still target old commits.

Safe execution order:

1. stop the backup daemon and prove no `gh release upload`/`git push` is active;
2. create and locally verify one clean closed-set snapshot commit;
3. retain an offline bundle of the old repository solely for rollback/audit;
4. recreate `main` from the clean snapshot, excluding the historical vendor
   paths, then rerun the all-history secret/license scanner;
5. force-push only `main` with `--force-with-lease` after explicit owner review;
6. recreate/rebind affected Release tags and metadata to the clean snapshot;
7. fresh-clone, verify the closed manifest and restore at least current top-1
   plus rolling resume before restarting the daemon.

Until that controlled transaction is approved and completed, ordinary new
snapshots may continue, but the public history should be described as
`HEAD cleaned; historical third-party residue pending controlled rewrite`.

Re-run command:

```bash
python scripts/audit_public_git_history.py \
  --root /root/autodl-tmp/srsc_lite_v12_github
```
