"""Archive/container helpers for IO modules.

Boundary:
- io helper: how bytes/files/archives are written and extracted
- not generic serialization policy
- not responsible for backend projection mechanics

v2 archives are zip+deflate(L1): inner parquet (zstd) and zarr (blosc) blobs
are already compressed, but text-heavy outliers (history.parquet, JSON blobs,
manifest) still compress significantly. Deflate at level 1 reclaims ~50% of
size at negligible CPU vs ZIP_STORED. zip is also random-access by member,
which lets future lazy loaders skip the extract step entirely.
"""

from pathlib import Path
import zipfile


def _write_archive(src_dir: Path, outfile: Path):
    src_dir = Path(src_dir)
    root_name = src_dir.name
    with zipfile.ZipFile(
        outfile, mode='w', compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as zf:
        # Walk deterministically so archive content order is stable.
        for entry in sorted(src_dir.rglob('*')):
            rel = entry.relative_to(src_dir)
            arcname = f'{root_name}/{rel.as_posix()}'
            if entry.is_dir():
                # Preserve empty directories so the extracted tree matches the source.
                if not any(entry.iterdir()):
                    zf.writestr(arcname + '/', b'')
                continue
            zf.write(entry, arcname=arcname)


def _read_archive(infile: Path, tmpdir: Path) -> Path:
    tmpdir = Path(tmpdir)
    with zipfile.ZipFile(infile, mode='r') as zf:
        zf.extractall(tmpdir)

    roots = [p for p in tmpdir.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise ValueError('Invalid annnet archive: expected single root directory')
    return roots[0]
