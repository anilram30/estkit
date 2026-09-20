#!/usr/bin/env python3
# Copyright 2026 Sreeram Anil
# SPDX-License-Identifier: Apache-2.0
"""Insert (idempotently) the copyright / SPDX header into every source, script, LaTeX and
Markdown file of the repository.

    python3 scripts/add_license_headers.py [--check] [root]

Code (.hpp .cpp .h .c .py .sh .cmake CMakeLists.txt) -> Apache-2.0
Report material (.tex .bib .md)                       -> CC-BY-4.0
Generated CSV data carry no header (it would break parsers); they are covered by results/README.md.
With --check the script only reports files without a header and exits non-zero if any.
"""
import os, sys, re

OWNER = 'Copyright 2026 Sreeram Anil'
CODE = 'Apache-2.0'
DOCS = 'CC-BY-4.0'
SKIP_DIRS = {'.git', 'build', 'site', '__pycache__', 'figures', 'ts', 'validation', 'node_modules'}
SKIP_FILES = {'LICENSE', 'NOTICE', 'LICENSE-CC-BY-4.0.txt'}


def header_for(path):
    name = os.path.basename(path)
    ext = os.path.splitext(name)[1]
    if ext in ('.hpp', '.cpp', '.h', '.c', '.inl'):
        return ['// ' + OWNER, '// SPDX-License-Identifier: ' + CODE], 'slash'
    if ext in ('.py', '.sh', '.cmake') or name == 'CMakeLists.txt':
        return ['# ' + OWNER, '# SPDX-License-Identifier: ' + CODE], 'hash'
    if ext in ('.tex', '.bib', '.sty'):
        return ['% ' + OWNER, '% SPDX-License-Identifier: ' + DOCS], 'percent'
    if ext in ('.md',):
        return ['<!-- ' + OWNER + ' — SPDX-License-Identifier: ' + DOCS + ' -->'], 'html'
    if ext in ('.yml', '.yaml'):
        return ['# ' + OWNER, '# SPDX-License-Identifier: ' + CODE], 'hash'
    if ext in ('.html', '.js', '.css') and 'site' in path.split(os.sep):
        return None, None   # generated site: header emitted by the generator
    return None, None


def process(path, check):
    lines, style = header_for(path)
    if lines is None:
        return None
    with open(path, encoding='utf-8') as f:
        s = f.read()
    if 'SPDX-License-Identifier' in s[:600]:
        return True
    if check:
        return False
    block = '\n'.join(lines) + '\n'
    if s.startswith('#!'):
        first, rest = s.split('\n', 1)
        s = first + '\n' + block + rest
    elif s.startswith('\\documentclass') or style == 'percent':
        s = block + s
    else:
        s = block + s
    with open(path, 'w', encoding='utf-8') as f:
        f.write(s)
    return True


if __name__ == '__main__':
    check = '--check' in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    root = args[0] if args else '.'
    missing, done = [], 0
    for d, dirs, files in os.walk(root):
        # also skip local build trees and install prefixes (build*/, _prefix/, __pycache__/):
        # they hold generated files, some of which (CMake export scripts) carry no header.
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not x.startswith('build') and not x.startswith('_')]
        for fn in files:
            if fn in SKIP_FILES:
                continue
            p = os.path.join(d, fn)
            r = process(p, check)
            if r is None:
                continue
            done += 1
            if r is False:
                missing.append(p)
    if check:
        print(f'{done} files checked, {len(missing)} without header')
        for m in missing:
            print('  missing:', m)
        sys.exit(1 if missing else 0)
    print(f'{done} files carry the licence header')
