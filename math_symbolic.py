"""Subprocess entry point for time-bounded symbolic math verification on Windows."""
import json
import re
import sys

from math_verify import LatexExtractionConfig, parse, verify


def parsed(value):
    # An unescaped percent can start a LaTeX comment and drop the scale.
    value = re.sub(r"(\d+(?:\.\d+)?)\s*\\?%", r"\\left(\\frac{\1}{100}\\right)", value)
    return parse("$" + value + "$", extraction_config=[LatexExtractionConfig()], parsing_timeout=None)


if __name__ == "__main__":
    gold, pred = json.loads(sys.stdin.buffer.read())
    print(json.dumps(bool(verify(parsed(gold), parsed(pred), timeout_seconds=None))))
