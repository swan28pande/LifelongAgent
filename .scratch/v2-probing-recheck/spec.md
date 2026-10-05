# Updated v2 dataset verification

Check the current `datasets/v2/` against the completed audit in `.research/v2-probing-audit/`. Determine which documented scheduling, reference, observation, citation, coverage, and source issues were corrected, remain present, or disappeared only because different questions were sampled. Inspect all current monthly question instances for regressions. Do not edit datasets, generator, agent, or runner code, and do not run paid model calls.

Question delivery follows the accepted `setup_2/` protocol: original probe dates and wording are preserved on repeats, with cumulative monthly memory. Distinguish question-file problems from the earlier legacy-loader limitation, which has already been addressed by `setup_2/`.
