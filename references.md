# References

Bibliography for the causal feature steering project. Organized by relevance.

## Direct prior work (the gap this project addresses)

**Briggs et al., 2026.** "Genome modeling and design across all domains of life with Evo 2." *Nature.*
The Evo 2 paper. 7B-param genomic foundation model trained on 9.3T nucleotides across all domains of life. Introduces the model we steer.

**Goodfire, 2025.** "Interpreting Evo 2." Technical Report.
Trains BatchTopK SAEs on Evo 2 internals, discovers interpretable biological features (coding sequences, protein secondary structure, tRNA motifs). Provides the SAE weights this project uses (`Goodfire/Evo-2-Layer-26-Mixed`, HuggingFace, MIT license). **Their work is descriptive; ours is functional.**

**Markov Bio, 2026.** "Through a Glass Darkly: Mechanistic Interpretability as the Bridge to End-to-End Biology." Blog/preprint.
Argues that the path from interpretability to utility requires *causal* interventions, not just descriptive feature identification. The motivating critique for this project.

**Zou, 2026.** *TIME* commentary on bio-FM interpretability.
Argues current bio-FM interpretability rediscovers known concepts without generating new biological insight. The "so what" challenge this project answers.

## SAE methodology

**Bricken et al., 2023.** "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning." Anthropic Transformer Circuits Thread.
Foundational SAE-for-interpretability paper. BatchTopK (used by Goodfire) is a successor variant.

**Abdel-Latif et al., 2026.** "Comprehensive feature atlases for single-cell foundation models." Preprint.
Adjacent work on SAE feature atlases for scGPT. Methodologically relevant; biologically orthogonal (cell-scale vs DNA-scale).

## Variant effect prediction (baselines + reward signal)

**Cheng et al., 2023.** "Accurate proteome-wide missense variant effect prediction with AlphaMissense." *Science.*
Provides the fast reward signal for the steering loop. Precomputed scores on Zenodo.

**Rentzsch et al., 2019.** "CADD: predicting the deleteriousness of variants throughout the human genome." *Nucleic Acids Research.*
Complement to AlphaMissense in the fast reward. Different feature basis; helps avoid Goodharting on a single predictor.

## Ground truth (held-out eval)

**Findlay et al., 2018.** "Accurate classification of BRCA1 variants with saturation genome editing." *Nature* 562:217–222.
Experimental functional effects for ~4,000 single-nucleotide variants across BRCA1's critical exons. **The MAVE ground truth for this project.** Available via MaveDB.

## Optimization (the GP policy)

**Frazier, 2018.** "A Tutorial on Bayesian Optimization." arXiv:1807.02811.
Reference for GP + Expected Improvement setup.

**Balandat et al., 2020.** "BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization." NeurIPS.
The optimization library used. Analytic EI, GPU-native acquisition.

## Genomic foundation models (context)

**Nguyen et al., 2024.** "Sequence modeling and design from molecular to genome scale with Evo." *Science.*
Evo 1, predecessor to Evo 2. Useful for understanding the architectural lineage.

## Adjacent prior context (from earlier project iterations)

These were considered for an earlier multi-scale chaining approach (BioChain) but aren't directly cited in the current proposal. Kept here for reference.

- **Wang et al., 2023.** "BioBRIDGE: Bridging Biomedical Foundation Models via Knowledge Graphs." ICLR 2024.
- **Xpressor, 2025.** bioRxiv. Cross-scale biological reasoning.
- **GenBio AI, 2024.** "AIDO: An AI-Driven Digital Organism." arXiv.
- **Cui et al., 2024.** "scGPT: toward building a foundation model for single-cell multi-omics." *Nature Methods.*

## Datasets

- **ClinVar.** NCBI Clinical Variation database. Filter: review status ≥ 2 stars, BRCA1/TP53 coding variants. Updated monthly.
- **MaveDB.** Repository for multiplexed assays of variant effect, including the Findlay et al. BRCA1 SGE data.

## Project doc

- `proposal.pdf` — the 1-page CS 153 proposal. Scope ground truth.