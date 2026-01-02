import tarfile
from pathlib import Path


def _write_archive(src_dir: Path, outfile: Path):
    with tarfile.open(outfile, "w:gz") as tar:
        tar.add(src_dir, arcname=src_dir.name)


def _read_archive(infile: Path, tmpdir: Path) -> Path:
    with tarfile.open(infile, "r:gz") as tar:
        tar.extractall(tmpdir)

    roots = list(tmpdir.iterdir())
    if len(roots) != 1:
        raise ValueError("Invalid annnet archive: expected single root directory")
    return roots[0]
