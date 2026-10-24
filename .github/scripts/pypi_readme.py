"""
Rewrite README.md for PyPI, which cannot resolve repository-relative paths.

Images become raw.githubusercontent.com URLs and other files github.com blob
URLs, both pinned to the given git ref so the PyPI page always shows the
figures of that release. In-page anchors point at the README on GitHub.

    python .github/scripts/pypi_readme.py README.md v1.2.1
"""
import re
import sys

REPO = 'rasidi3112/quantum-simulation'
IMAGE_EXT = ('.png', '.gif', '.jpg', '.jpeg', '.svg', '.webp')


def absolute(path: str, ref: str) -> str:
    if path.startswith(('http://', 'https://', 'mailto:')):
        return path
    if path.startswith('#'):
        return f'https://github.com/{REPO}/blob/{ref}/README.md{path}'
    target, _, anchor = path.partition('#')
    if target.lower().endswith(IMAGE_EXT):
        url = f'https://raw.githubusercontent.com/{REPO}/{ref}/{target}'
    else:
        url = f'https://github.com/{REPO}/blob/{ref}/{target}'
    return url + (f'#{anchor}' if anchor else '')


def rewrite(text: str, ref: str) -> str:
    text = re.sub(r'\]\(([^)\s]+)\)', lambda m: f']({absolute(m.group(1), ref)})', text)
    return re.sub(r'(src|href)="([^"]+)"', lambda m: f'{m.group(1)}="{absolute(m.group(2), ref)}"', text)


if __name__ == '__main__':
    path, ref = sys.argv[1], sys.argv[2]
    with open(path, encoding='utf-8') as f:
        text = f.read()
    with open(path, 'w', encoding='utf-8') as f:
        f.write(rewrite(text, ref))
