# Baseline codebases for setup_2

Status: complete

Prepare separate TiMem, NaiveRAG, and DirectPrompting codebases under
`setup_2/baselines/`, reusing existing local implementations. The user confirmed
that DirectPrompting means the existing full-context method.

Keep method behavior and shared benchmark settings intact. Record source paths,
dependencies, and limitations that matter before benchmarking. Validate copied
source and imports without making model calls or starting database services.

This stage does not integrate baseline selection into the monthly runner, execute
benchmarks, repair existing baseline algorithms, or change agents or datasets.

## Acceptance

- Each requested baseline has a separate folder containing its implementation.
- TiMem includes its local source, configuration, prompts, dependencies, and license.
- NaiveRAG includes the original baseline and its existing benchmark adapter.
- DirectPrompting uses the existing FullContext benchmark implementation.
- The copied adapters can be imported from their new locations.
- Existing input and implementation files remain unchanged.

## Completion

Prepared all three folders with copied implementations, dependency files, and
documentation. Adapter imports and syntax checks passed; NaiveRAG and full context
passed offline incremental ingestion/answer checks. No model calls or service
launches were made. Details: `copy_manifest.json` and `validation.json` here, and
`setup_2/baselines/README.md`.
