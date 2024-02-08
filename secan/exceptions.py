class SectionUnstableError(Exception):
    """Exception raised when a section cannot support the given loads."""
    pass

class ConvergenceError(Exception):
    """Exception raised when the algorithm fails to converge to a solution."""
    pass