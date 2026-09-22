"""The compile gate: a refused program never becomes an answer (evaluate.py).

evaluate.py routes every prediction through a cohortc binary that either compiles
the program or refuses it. These contract checks use stub binaries to prove the
gate's semantics without requiring cohortc to be built: a non-zero exit is
recorded as a compile failure (with the compiler's message), never scored.
"""
import tempfile
import unittest
from pathlib import Path

from evaluate import score

ACCEPT_STUB = "#!/bin/sh\ncat > /dev/null\nexit 0\n"
REFUSE_STUB = ("#!/bin/sh\ncat > /dev/null\n"
               "echo 'error: unknown measure `rank`' >&2\nexit 1\n")


def make_stub(directory: Path, body: str) -> str:
    stub = directory / "cohortc-stub.sh"
    stub.write_text(body, encoding="utf-8")
    stub.chmod(0o755)
    return str(stub)


class CompileGateTests(unittest.TestCase):
    def test_refused_program_is_counted_as_failure_never_scored(self):
        with tempfile.TemporaryDirectory() as d:
            binary = make_stub(Path(d), REFUSE_STUB)
            preds = [{"pred": "cohort psychiatrists@nucc\nmeasure count",
                      "output": "cohort kol\nmeasure count"}]
            result = score(preds, binary, "unused-registry.yaml")
            self.assertEqual(result["compiles"], 0)
            self.assertEqual(
                result["failures"]["error: unknown measure `rank`"], 1)

    def test_compiling_program_passes_the_gate(self):
        with tempfile.TemporaryDirectory() as d:
            binary = make_stub(Path(d), ACCEPT_STUB)
            preds = [{"pred": "cohort kol\nmeasure count",
                      "output": "cohort kol\nmeasure count"}]
            result = score(preds, binary, "unused-registry.yaml")
            self.assertEqual(result["compiles"], 1)
            self.assertEqual(result["cohort_ok"], 1)
            self.assertEqual(result["exact"], 1)
            self.assertEqual(len(result["failures"]), 0)


if __name__ == "__main__":
    unittest.main()
