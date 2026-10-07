# Session harnesses — the strata_dirigo performance session, 2026-10-07

These 43 scripts are the drivers that produced every number in `../../PERFORMANCE-FINAL-SURVEY.md` and
`../../MOTHBALL-REPORT.md`. They were written in `/home/bob/step4/` during the session and are copied here
because scripts left in `~` do not survive to the next session — the lesson the `code-audit` skill records as
"evidence that outlives the session".

**They expect `/home/bob/step4/` as their working directory** and use absolute paths (`/home/bob/vkbuild-vulkan/`,
`/home/bob/step4/bins/`, `/tmp/b70.lock`). To re-run one:

```bash
mkdir -p /home/bob/step4 && cp session/*.sh /home/bob/step4/ && cd /home/bob/step4 && bash <script>
```

## What each group does

| script(s) | what it does |
|---|---|
| `run.sh` | the arm runner — `<name> <tokens> [extra args]`; one engine run, fully logged |
| `build_product.sh` | rebuilds the product and stamps its identity (the stamp is what a claim is tied to) |
| `gate_run.sh` | invokes the port's gate **with no arguments** — see the trap in `../../MOTHBALL-REPORT.md` §10 |
| `direct_upload_ab.sh`, `xfer_attrib.sh` | the `STRATA_VK_DIRECT_UPLOAD` A/B and the upload/dispatch accounting |
| `gemm_{tiled,ku,cm}_ab.sh`, `prefill_{chunk,reg,d1,phase,only_dispstat}.sh` | the prefill kernel A/Bs and the phase/dispatch attribution |
| `hostgroup_profile.sh` | the privileged perf attach for the host-grouping phase |
| `hipcontrol.sh`, `hipcontrol_count.sh`, `hipbuild.sh` | the HIP control run on z820b and its `LD_PRELOAD` launch counter |
| `selftest_*.sh`, `assert_{sha,diff}.sh`, `repro_test.sh` | harness self-tests — **run these first if you doubt a number** |
| `nan_*.sh`, `sweep8.sh`, `confirm199.sh`, `task1_arms.sh`, `task2_verify.sh` | the all-zeros investigation and the family sweeps |

## Provenance

Copies of `/home/bob/step4/*.sh` as of 2026-10-07 19:45, HEAD `c123372`. Checked for credentials before copying:
the only pattern hits were the words "tokens" and "passwordless", and a search for actual `KEY=`-style
assignments found none.
