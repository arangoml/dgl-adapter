class ADBDGLError(Exception):
    """Base class for all exceptions in adbdgl-adapter."""


class ADBDGLValidationError(ADBDGLError, TypeError):
    """Base class for errors originating from adbdgl-adapter user input validation."""


##################
#   Metagraphs   #
##################


class ADBMetagraphError(ADBDGLValidationError):
    """Invalid ArangoDB Metagraph value"""


class DGLMetagraphError(ADBDGLValidationError):
    """Invalid DGL Metagraph value"""
