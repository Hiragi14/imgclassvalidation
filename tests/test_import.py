"""Basic tests for the package."""


def test_import_package() -> None:
    import imgclassvalidation

    assert imgclassvalidation is not None


def test_version_exists() -> None:
    import imgclassvalidation

    assert hasattr(imgclassvalidation, "__version__")
    assert isinstance(imgclassvalidation.__version__, str)
