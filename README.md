# AbsorbBench

per-item engagement coefficient ψ on observational recommender-system logs.
75M (u, i, t) interactions, short-form video + news/text.

## What is ψ

ψ(u, i, t) = c · (1 + R̃_norm) / K(i)

- c = engagement-time fraction (T_engaged / T_total), in [0, 1]
- R̃_norm = normalised reflective-act count (per-modality 90th-percentile cap), in [0, 1]
- K(i) = per-modality content entropy in bits, floored at 1

ψ is bounded in [0, 2]. derivation + soundness audit are in the OSF preregistration linked at the bottom.

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

EB-NeRD was originally going to be the danish-news shard but got dropped after K-validation failed (5 different K_text formulations all failed convergent + LIX-anchor). see preregistration §3 for the drop rule.

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

F6 (cold-start informativeness ramp, added 2026-05-02) reports the split-half Spearman ρ on ψᵢ within encounter-count buckets and the intra-class correlation ICC(1, k) of the ψᵢ mean as a function of k encounters. Closes the "n=1 is just noise" question for cold-start items: on Tenrec QK-article, ICC(1, 1) = 0.81 and split-half ρ exceeds 0.67 from n = 2.

F1-F6 are reported as stability-and-overlap analyses, not as falsifiability gates that bind a construct claim. The construct claim was withdrawn pre-release per the binding pre-registration; ψᵢ ships as a per-item engagement coefficient with documented stability and documented overlap with popularity.

## Status

- corpus build: done
- invariant tests: passing
- baselines: done. final 3-seed numbers from the H100 run land in the paper.
- paper: in progress, deadline May 6
- zenodo deposit: todo

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
