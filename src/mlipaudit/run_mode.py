import enum


class RunMode(enum.Enum):
    """Enum for the mode of a benchmark run.

    Attributes:
        DEV: Very minimal and fast. Just meant for testing.
        FAST: For some long-running benchmarks, a limited set of test cases is run to
              decrease overall runtime. For most benchmarks, this is not different
              from the standard case.
        STANDARD: Complete run of all benchmark cases.

    """

    DEV = "dev"
    FAST = "fast"
    STANDARD = "standard"
