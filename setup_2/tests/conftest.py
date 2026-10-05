"""Keep all offline test artifacts within setup_2, including pytest's defaults."""

import os
from pathlib import Path
import tempfile


def pytest_configure(config):
    temporary = Path(__file__).resolve().parents[1] / 'tmp'
    temporary.mkdir(exist_ok=True)
    os.environ['TMPDIR'] = str(temporary)
    tempfile.tempdir = str(temporary)
    if config.option.basetemp is None:
        config.option.basetemp = str(temporary / 'pytest')
    for key in ('LANGSMITH_TRACING', 'LANGSMITH_TRACING_V2', 'LANGCHAIN_TRACING', 'LANGCHAIN_TRACING_V2'):
        os.environ[key] = 'false'
