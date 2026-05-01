"""Package metadata helpers."""

from __future__ import annotations

import re
from html import escape
from typing import Any
from pathlib import Path
from importlib import metadata as importlib_metadata
from dataclasses import dataclass
from urllib.parse import urlparse
import urllib.request

from .optional_components import (
    IO_MODULES,
    PLOT_BACKENDS,
    GRAPH_BACKENDS,
    DATAFRAME_BACKENDS,
    component_names,
    component_status,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

__all__ = [
    'get_latest_version',
    'info',
    '__title__',
    '__version__',
    '__author__',
    '__authors__',
    '__maintainers__',
    '__license__',
]

_FALLBACK_VERSION = '0.1.0'
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT_PATH = _PROJECT_ROOT / 'pyproject.toml'
_OPTIONAL_BUNDLES = {'io', 'backends', 'plot', 'bio', 'storage', 'all', 'dev'}

_CHIP_STYLE = (
    'display:inline-flex;align-items:center;gap:.28rem;'
    'padding:.12rem .45rem;margin:.08rem .35rem .08rem 0;'
    'border:1px solid #d0d7de;border-radius:999px;'
)


def _normalize_people(entries: Any) -> list[str]:
    if not isinstance(entries, list):
        return []

    people = []
    for entry in entries:
        if isinstance(entry, str):
            people.append(entry)
        elif isinstance(entry, dict):
            name, email = entry.get('name'), entry.get('email')
            if name and email:
                people.append(f'{name} <{email}>')
            elif name or email:
                people.append(str(name or email))
    return people


def _normalize_license(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get('text') or value.get('file')
    return None


def _metadata_from_pyproject() -> dict[str, Any]:
    if not _PYPROJECT_PATH.exists():
        return {}

    with _PYPROJECT_PATH.open('rb') as fh:
        pyproject = tomllib.load(fh)

    project = pyproject.get('project', {})
    authors = _normalize_people(project.get('authors'))
    maintainers = _normalize_people(project.get('maintainers'))

    return _drop_none(
        {
            'name': project.get('name'),
            'version': project.get('version'),
            'authors': authors,
            'author': ', '.join(authors) or None,
            'maintainers': maintainers,
            'license': _normalize_license(project.get('license')),
            'urls': project.get('urls', {}),
            'full_metadata': pyproject,
        }
    )


def _metadata_from_installed_package() -> dict[str, Any]:
    try:
        pkg_meta = importlib_metadata.metadata('annnet')
    except importlib_metadata.PackageNotFoundError:
        return {}

    def fields(*names: str) -> list[str]:
        return list(dict.fromkeys(v for name in names if (v := pkg_meta.get(name))))

    authors = fields('Author', 'Author-email')
    maintainers = fields('Maintainer', 'Maintainer-email')

    return _drop_none(
        {
            'name': pkg_meta.get('Name'),
            'version': pkg_meta.get('Version'),
            'authors': authors,
            'author': ', '.join(authors) or None,
            'maintainers': maintainers,
            'license': pkg_meta.get('License'),
            'summary': pkg_meta.get('Summary'),
            'full_metadata': dict(pkg_meta.items()),
        }
    )


def _drop_none(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if v is not None}


def _get_metadata() -> dict[str, Any]:
    meta = _metadata_from_pyproject() or _metadata_from_installed_package()

    meta.setdefault('name', 'annnet')
    meta.setdefault('version', _FALLBACK_VERSION)
    meta.setdefault('authors', [])
    meta.setdefault('maintainers', [])
    meta.setdefault('author', ', '.join(meta['authors']) or None)
    meta.setdefault('license', 'BSD-3-Clause')
    meta.setdefault('urls', {})

    return meta


def _optional_dependency_bundles(meta: dict[str, Any]) -> dict[str, list[str]]:
    optional = meta.get('full_metadata', {}).get('project', {}).get('optional-dependencies', {})

    if not isinstance(optional, dict):
        return {}

    return {k: v for k, v in optional.items() if k in _OPTIONAL_BUNDLES}


def _author_links(authors: list[str]) -> str:
    rendered = []

    for author in authors:
        if '<' in author and author.endswith('>'):
            name, email = author.rsplit('<', 1)
            email = email[:-1].strip()
            rendered.append(
                f'{escape(name.strip())} '
                f'<a href="mailto:{escape(email)}" style="text-decoration:none" '
                f'title="{escape(email)}">&#9993;</a>'
            )
        else:
            rendered.append(escape(author))

    return ', '.join(rendered) or 'n/a'


def _component_message(values: dict[str, dict[str, str]]) -> str:
    if not values:
        return 'n/a'

    return '; '.join(
        f'{name}: {details["available"]}'
        f'{f" ({details['install']})" if details["available"] == "no" else ""}'
        for name, details in values.items()
    )


def _default_available(values: dict[str, dict[str, str]], backends: Any) -> str:
    return next(
        (
            name
            for name in component_names(backends)
            if values.get(name, {}).get('available') == 'yes'
        ),
        'none',
    )


def _chips(
    items: dict[str, bool] | dict[str, list[str]],
    *,
    icons: bool = True,
    titles: bool = False,
) -> str:
    if not items:
        return "<span style='color:#57606a'>none</span>"

    html = []
    for name, details in items.items():
        icon = ''
        title = ''
        ok = bool(details) if isinstance(details, bool) else True
        if icons:
            color = '#1a7f37' if ok else '#cf222e'
            symbol = '&#10003;' if ok else '&#10007;'
            icon = f"<span style='color:{color};font-weight:700'>{symbol}</span>"
        if titles:
            title = f' title="{escape(", ".join(details))}"'

        html.append(f"<span{title} style='{_CHIP_STYLE}'>{icon}<span>{escape(name)}</span></span>")

    return ''.join(html)


def _link(url: str) -> str:
    safe = escape(url)
    return f'<a href="{safe}">{safe}</a>'


def get_latest_version(
    url: str = 'https://raw.githubusercontent.com/saezlab/annnet/main/pyproject.toml',
    timeout: int = 5,
) -> str | None:
    """Fetch the latest version declared on the default branch."""

    if urlparse(url).scheme not in {'http', 'https'}:
        return None

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # nosec B310
            content = response.read().decode()
    except (OSError, UnicodeDecodeError):
        return None

    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    return match.group(1) if match else None


@dataclass(frozen=True)
class AnnNetInfo:
    metadata: dict[str, Any]
    graph_backends: dict[str, dict[str, str]]
    plot_backends: dict[str, dict[str, str]]
    tabular_backends: dict[str, dict[str, str]]
    io_modules: dict[str, dict[str, str]]

    def _text_rows(self) -> list[tuple[str, str]]:
        urls = self.metadata.get('urls', {})
        version = self.metadata.get('version', _FALLBACK_VERSION)
        license_name = self.metadata.get('license', 'BSD-3-Clause')

        rows = [('Installed version', f'v{version}'), ('License', license_name)]
        if self.metadata.get('author'):
            rows.append(('Authors', self.metadata['author']))

        rows.extend(
            [
                ('Available graph backends', _component_message(self.graph_backends)),
                ('Available plot backends', _component_message(self.plot_backends)),
                ('Available tabular backends', _component_message(self.tabular_backends)),
                ('Available I/O modules', _component_message(self.io_modules)),
            ]
        )

        for key in ('Repository', 'Documentation'):
            if urls.get(key):
                rows.append((key, urls[key]))
        rows.append(('Installed path', str(_PROJECT_ROOT)))
        return rows

    def __str__(self) -> str:
        return '\n'.join(f'{title}: {message}' for title, message in self._text_rows())

    __repr__ = __str__

    def _repr_html_(self) -> str:
        version = self.metadata.get('version', _FALLBACK_VERSION)
        urls = self.metadata.get('urls', {})
        optional = _optional_dependency_bundles(self.metadata)

        graph_default = _default_available(self.graph_backends, GRAPH_BACKENDS)
        plot_default = _default_available(self.plot_backends, PLOT_BACKENDS)

        rows = [
            ('Version', f'v{escape(str(version))}'),
            ('License', escape(str(self.metadata.get('license', 'BSD-3-Clause')))),
            ('Authors', _author_links(self.metadata.get('authors', []))),
            ('Repository', _link(urls['Repository'])) if urls.get('Repository') else None,
            ('Documentation', _link(urls['Documentation'])) if urls.get('Documentation') else None,
            ('Installed path', escape(str(_PROJECT_ROOT))),
            (
                'Default graph backend',
                _chips({graph_default: graph_default != 'none'}, icons=False),
            ),
            ('Default plot backend', _chips({plot_default: plot_default != 'none'}, icons=False)),
            (
                'Graph backends',
                _chips({k: v['available'] == 'yes' for k, v in self.graph_backends.items()}),
            ),
            (
                'Plot backends',
                _chips({k: v['available'] == 'yes' for k, v in self.plot_backends.items()}),
            ),
            (
                'Tabular data backends',
                _chips({k: v['available'] == 'yes' for k, v in self.tabular_backends.items()}),
            ),
            (
                'I/O modules',
                _chips({k: v['available'] == 'yes' for k, v in self.io_modules.items()}),
            ),
            ('Installable bundles', _chips(optional, icons=False, titles=True)),
        ]

        table_rows = ''.join(
            '<tr>'
            "<th style='text-align:right;vertical-align:top;padding:.2rem .9rem .2rem 0;"
            f"white-space:nowrap;width:10.5rem;'>{escape(title)}</th>"
            f"<td style='text-align:left;padding:.2rem 0'>{value}</td>"
            '</tr>'
            for row in rows
            if row is not None
            for title, value in [row]
        )

        return (
            '<div style=\'font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;'
            'border:1px solid #d0d7de;border-radius:12px;padding:1rem 1.1rem;'
            "max-width:820px;background:#fff;'>"
            "<div style='margin-bottom:.6rem;'>"
            "<div style='font-size:1.05rem;font-weight:700;'>annnet</div>"
            f"<div style='font-size:.92rem;color:#57606a;'>"
            f'Environment summary for annnet v{escape(str(version))}</div>'
            '</div>'
            "<table style='border-collapse:collapse;width:100%;'><tbody>"
            f'{table_rows}'
            '</tbody></table></div>'
        )

    def _mime_(self) -> tuple[str, str]:
        return 'text/html', self._repr_html_()

    def to_html(self) -> str:
        return self._repr_html_()


metadata = _get_metadata()

__title__ = metadata['name']
__version__ = metadata['version']
__author__ = metadata.get('author')
__authors__ = metadata['authors']
__maintainers__ = metadata['maintainers']
__license__ = metadata['license']


def info() -> AnnNetInfo:
    """Return a human-readable package component summary."""

    return AnnNetInfo(
        metadata=_get_metadata(),
        graph_backends=component_status(GRAPH_BACKENDS),
        plot_backends=component_status(PLOT_BACKENDS),
        tabular_backends=component_status(DATAFRAME_BACKENDS),
        io_modules=component_status(IO_MODULES),
    )
