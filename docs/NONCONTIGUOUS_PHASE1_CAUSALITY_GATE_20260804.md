# Phase 1 Causality Gate

Run date: 2026-08-04. This gate ran before any A-B-A history-injection mode.

## Matched runs

All runs used seed 101, `configs/self_forcing_dmd.yaml`, the EMA Self-Forcing
checkpoint, the prepared 7.5-second A/B/A2 prompt, 480x832, four denoising
steps, and 30 latent frames. The two opt-in controls used source blocks 3 and
6, target block 8, `baseline` mode, and retrieval count 1. No history was
injected in any run.

```bash
conda run -n wan python /tmp/run_infinity_rope_causality_gate.py
```

The runner called `torch.set_grad_enabled(False)`, exactly as `inference.py`
does. It recorded after every clean pass, before MP4 encoding: SHA-256 of the
complete clean latent tensor; SHA-256 of every persistent K/V tensor plus
cache indices; and exact max absolute differences against normal inference.
It stored normal-reference snapshots only in a temporary directory and removed
them after comparison. Raw metrics remain at
`outputs/noncontiguous_phase1_causality_gate/metrics.json`.

The first standalone-runner attempt omitted that inference-mode setting and
OOMed at block 2 from retained autograd activations. It did not produce a gate
result. After matching the live inference setting, all three recorded runs
completed with the unchanged generation configuration above.

| Run | Runtime | Raw decoded RGB SHA-256 | Max abs RGB difference vs normal |
| --- | ---: | --- | ---: |
| normal, noncontiguous disabled | 29.229 s | `cd334157d2f34a5a833b68ab5624d5d9819df3586e7f6b4767b09f039624e0c4` | 0.0 |
| opt-in baseline, capture/offload, no injection | 34.690 s | `cd334157d2f34a5a833b68ab5624d5d9819df3586e7f6b4767b09f039624e0c4` | 0.0 |
| repeated opt-in baseline | 28.805 s | `cd334157d2f34a5a833b68ab5624d5d9819df3586e7f6b4767b09f039624e0c4` | 0.0 |

## Per-clean-pass evidence

For every block below, normal, opt-in baseline, and repeated opt-in baseline
have identical hashes; both opt-in runs have clean-latent max abs `0.0`,
persistent-K/V max abs `0.0`, and identical persistent-cache indices against
normal.

| Block | Clean latent SHA-256 | Persistent K/V SHA-256 |
| ---: | --- | --- |
| 1 | `a554dcc2422d42215ca652359bffbc855a37ab1a3f321b2c849dcc6abbc4d10b` | `bd150f4d88eabc47f1770c27f59c26ab39a6ebc7d9d77b3c30ff9bfa34cfe5b2` |
| 2 | `e7b2a92c9660b3012ebeda2f31af1dbd41b1028400ae747c49f7e0b6c8a13309` | `3aaf55e955a20e2f5962cb78ba1ab2feec7b8a4a0585f6497fe7b1b228e5901b` |
| 3 | `4803894a0b4c4d623a55e7689b6b7be74fda274901283a1e6ba9378ea3f5a96b` | `2377131be75e27d8f8bb8045ca90e3c1501c834b2b75fcf19571ee64720efe9a` |
| 4 | `0f256843290880666ae2bdeb7324694358a19c4bf7978c6b99e34a346ad3723e` | `959b18e0c9f1da305fe732a7327b59027933ebee07212c26b51a387ccc955608` |
| 5 | `fa78406fb25edd3267b261776e327d84aa32ff5f3af9fc0c75ac925e693826c6` | `473ead6f79343ca9e3746b4ca0913351bfc4519e4bb8e7bdf62981d14dc205c3` |
| 6 | `ff575085c05b61c572d29199f88d2f7f1f8058efb807c033e47ea35bf1a5eca1` | `ad662bbb8cfefdc63faab1b5bd9b68323aac3443e4f267cd5bd14ef838ff2389` |
| 7 | `d7fbf3e10f8dcaecb6a09628b076b997956efbacab672500d763ee1da712edad` | `45f80b2f1d8a277ba14ba13cfc2bfd81e6f6205fa27e70a0809100579e9de49e` |
| 8 | `663a40389a991e70e15702fe39d25a374aa195354db359a0f45153b16bfcfa24` | `c3016144d2f20e74b3b02697abc06a8a027123f9db3d83afe23d78cb92b09c78` |
| 9 | `6988f49ea984835bddc63d2cc885884fa086bac2e5fd6fb6e1b66483eefd4338` | `fea19736b4c5086b6684c6f14f6314a9956b29b91d5a04b092484e2b7b5f4aba` |
| 10 | `a7d329da3eb986103e3c3554983c32e0aea749998f2b125722c2052b97027ddc` | `a59b98dbf290bea726e2d653566db58e4c4aad30c96b0500d5f4e7f668356359` |

## Result

No divergence occurred, so there is no first divergent block or tensor to fix.
The capture path stores a detached GPU clone of raw clean K/V before
`capture_clean_kv_to_cpu` makes its independent CPU copy. Selection uses the
private Python `Random` instance; the focused regression test verifies that it
does not advance Torch's global RNG.
