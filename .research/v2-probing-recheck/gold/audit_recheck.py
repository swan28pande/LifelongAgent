#!/usr/bin/env python3
"""Reuse baseline semantic audit with declared adaptations for resampled rows.

No input is edited and no model is called. The baseline's report is deliberately
disabled because its prose hardcodes prior counts/cases. Its semantic checks and
machine evidence remain intact except that absence of a rejected counterfactual
is allowed (new samples may have coincident first values but unique boundaries),
and hypothetical starts after the original probe are excluded as witnesses.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


class UpdateAssertions(ast.NodeTransformer):
    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "rejected"):
            before_probe = ast.parse(
                "alternatives = [a for a in alternatives if a['start_day'] <= viewpoint]"
            ).body[0]
            return [before_probe, node]
        return node

    def visit_Assert(self, node):
        value = ast.unparse(node.test)
        if value == "rejected":
            return None
        if value == "total_rows == 1000 and len(all_coverage) == 511":
            return ast.parse("assert total_rows == 1000").body[0]
        if value == "len({r['id'] for r in all_coverage}) == 511":
            return ast.parse("assert len({r['id'] for r in all_coverage}) == len(all_coverage)").body[0]
        return node

    def visit_Expr(self, node):
        if (isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "finding"
                and node.value.args
                and isinstance(node.value.args[0], ast.Constant)
                and node.value.args[0].value == "non_unique_change_date_rejected_by_grader"):
            return ast.If(test=ast.Name(id="rejected", ctx=ast.Load()), body=[node], orelse=[])
        return node


def main():
    baseline = ROOT / ".research/v2-probing-audit/gold/audit_gold.py"
    namespace = runpy.run_path(str(baseline), run_name="baseline_gold_import")
    globals_ = namespace["main"].__globals__
    globals_.update(ROOT=ROOT, OUT=OUT, write_report=lambda *args: None)
    tree = UpdateAssertions().visit(ast.parse(inspect.getsource(namespace["main"])))
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(OUT / "audit_recheck.py") + ":adapted_baseline_main", "exec"), globals_)
    globals_["main"]()


if __name__ == "__main__":
    main()
