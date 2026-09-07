# Sources and licenses

## Code used directly

### Pathway public BDH toy implementation
- Source: https://github.com/pathwaycom/bdh
- Role: base BDH architecture and training reference.
- License: preserve the upstream license included in the repository.
- Team modifications: independent checkpoint training, state-extraction instrumentation, FastAPI layer, A/B experiment, pre/post-ReLU measurement, learning challenge, and evidence labeling.

## Repositories reviewed for interaction / visualization inspiration

The following repositories were reviewed as references. Do not claim code was incorporated unless a final-file audit confirms copied/adapted source.
- TensorFlow Playground
- BertViz
- OpenAI Sparse Autoencoder
- Neuronpedia

If any source code from these projects is ultimately copied into the submission, add the exact files/components and the applicable license obligations here before submission.

## Data
- Tiny Shakespeare / Karpathy char-rnn dataset, fetched by the public BDH toy training script.

## Model weights
- `bdh_trained.pt`: independently trained by the team from the public toy BDH architecture on Tiny Shakespeare. Not an official Pathway checkpoint.

## Research
1. Kosowski et al. (2025), arXiv:2509.26507.
2. Cunningham et al. (2023), arXiv:2309.08600.
3. Gao et al. (2024), arXiv:2406.04093.
4. Engdahl et al. (2026), arXiv:2608.09888.
