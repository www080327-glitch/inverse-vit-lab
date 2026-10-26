# C15 final 9000-step delivery

Read [HANDOFF_ZH.md](HANDOFF_ZH.md), [RESULT.json](RESULT.json), and [SOURCES.md](SOURCES.md).
[C15_DELIVERY.zip](C15_DELIVERY.zip) includes both the 9000-step study and the complete initial_1800/ stage.

From a clean checkout in this directory, install CPU torch==2.5.1 and requirements.txt, then run python run.py. Existing results/budget.json blocks silent retraining. For a new repeat, preserve these outputs and explicitly select a fresh C15_OUTPUT_DIR. The default config carries forward the initial-stage training cost when enforcing the 1800-second cumulative limit.

The source repository base directory also contains config_extended.json and finalize.py for hosted execution/packaging. All code/data definitions are included; no trained weights or raw datasets are included.

Portable interface: Head.sample(condition,n,generator) returns [B,n,d]. Accepts integer IDs or [B,condition_dim] continuous features. Use Head(...,condition_dim=h) for h-dimensional upstream features. Only the 4-dimensional one-hot conditions were trained in this study; vector interface checks do not validate real visual generation.
