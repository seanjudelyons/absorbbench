# AbsorbBench

per-item engagement coefficient ψ on observational recommender-system logs.
75M (u, i, t) interactions, short-form video + news/text.

## What is ψ

ψ(u, i, t) = c · (1 + R̃_norm) / S(i)

- c = engagement-time fraction (T_engaged / T_total), in [0, 1]
- R̃_norm = normalised reflective-act count (per-modality 90th-percentile cap), in [0, 1]
- S(i) = per-item categorical-metadata surprisal in bits, floored at 1

ψ is bounded in [0, 2]. derivation + soundness audit are in `paper/` (Appendix A, D1–D7).

KuaiRand-Pure ships two ψ variants: ψ_S0 under the single-tag marginal, and ψ_S3 under the joint marginal of (tag × log10-duration bucket). On KR-Pure ψ_S3 lifts ICC(1,1) from 0.22 to 0.67 and the canonical F4 cross-slice Pearson r from 0.679 to 0.879. Tenrec QK-article ships one ψ under the (cat_first, cat_second) joint; the read+reflect numerator is already at ICC(1,1)=0.79 there, so additional S-richness adds ≤+0.001.

## Layout

```
corpus_build/   parquet build pipeline (kuairand-pure, tenrec qk-article, + 3 appendix shards)
baselines/      trivial, popularity, mlp, sasrec, bert4rec, linucrl, falsifiability F1-F6
tests/          corpus invariant smoke tests
```

## Datasets

source datasets aren't bundled. you have to grab them:

- **KuaiRand-Pure** — https://kuairand.com/ (direct download, CC BY-SA 4.0)
- **Tenrec QK-article** — https://static.qq.com/qbs/Tenrec/ (manual application via portal; took me about a week)
- **KuaiRec** — https://kuairec.com/ (direct download)
- **Tenrec QK-video, QB-video** — same Tenrec portal as above

EB-NeRD was originally going to be the danish-news shard but got dropped after S-validation failed (5 different S_text formulations all failed convergent + LIX-anchor). see paper §3 for the drop rule.

music streaming was excluded by design — ψ presupposes foreground attention, which music doesn't satisfy.

## Build

```
pip install -r requirements.txt
python -m corpus_build.build --datasets-dir /path/to/your/datasets/
```

build pipeline produces sha256-stable parquet manifests. ~20 min on a laptop for the spine; ~40 min if you also build the appendix shards.

## Baselines

```
python -m baselines.popularity --corpus corpus/
python -m baselines.mlp_tabular --shard corpus/spine/kuairand_pure.parquet --split items --epochs 10
```

linucrl/sasrec/bert4rec take longer (gpu recommended for sasrec + bert4rec on tenrec).

falsifiability F1-F6 are cpu-only and run in a few minutes per shard:

```
python -m baselines.falsifiability_F1_user_half --shard corpus/spine/tenrec_qk_article.parquet
python -m baselines.falsifiability_F6_coldstart_ramp --shard corpus/spine/tenrec_qk_article.parquet
```

F6 (cold-start informativeness ramp) reports the split-half Spearman ρ on ψᵢ within encounter-count buckets and the intra-class correlation ICC(1, k) of the ψᵢ mean as a function of k encounters. On Tenrec QK-article, ICC(1, 1) = 0.81 and split-half ρ exceeds 0.67 from n = 2.

F7 (external behavioural anchor on KuaiRand-Pure) tests whether ψᵢ measured on the first 70% of the timeline predicts behavioural signals not used in the ψ formula on the held-out last 30%. Three of five anchors pass (comment_stay_time ρ=0.29, click rate ρ=0.44, long_view ρ=0.52); the corpus is not self-validating.

F1–F7 are stability, overlap, and external-anchor analyses, not falsifiability gates that bind a construct claim. ψᵢ ships as a per-item engagement coefficient with documented n=1 reliability, documented overlap with popularity, and documented predictive signal beyond its own arithmetic components.

## Status

- corpus build: done
- invariant tests: 8/8 passing
- baselines: done. final 3-seed numbers from the H100 run are in the paper (Table 4).
- paper: submitted to NeurIPS 2026 E&D Track (notification 2026-09-24).
- zenodo deposit: live at https://doi.org/10.5281/zenodo.19965136 (Track A: KuaiRand-Pure + KuaiRec).

## Notes / known issues

- the corpus is ~620 MB compressed (parquet + manifest). don't try to upload the tarball to google drive's web UI; it silently truncates >500 MB files. ask me how i know.
- determinism is set where supported but sasrec is non-deterministic across cuda versions even with `torch.use_deterministic_algorithms(True)`. seeds are logged.
- two languages (chinese for both spine subsets). western short-video data isn't in the spine because there's no public dataset with the requisite per-(u,i,t) signals.

## License

code in this repo: Apache 2.0 (LICENSE file).
the computed ψ column inherits the source dataset's license (CC BY-SA 4.0 for kuairand; custom non-commercial for tenrec).

## Cite

bibtex will be added after acceptance / arXiv posting. for now:

> Sean Lyons. AbsorbBench: per-item engagement on observational recommender logs. NeurIPS 2026 E&D Track (under review). 2026.
