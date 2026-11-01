"""Pin the tested runtime dependency closure without modifying training requirements."""
from pathlib import Path
from importlib import metadata
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
DIRECT = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl',
          'matplotlib', 'shap', 'streamlit', 'numba', 'llvmlite']


def main():
    pending = list(DIRECT)
    versions = {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in versions:
            continue
        dist = metadata.distribution(name)
        versions[name] = dist.version
        for text in dist.requires or []:
            requirement = Requirement(text)
            if requirement.marker and not requirement.marker.evaluate({'extra': ''}):
                continue
            pending.append(requirement.name)
    header = '# Tested Python 3.13 runtime dependency closure. Training requirements are frozen separately.\n'
    # colorama is Windows-only when used by click; harmless on Linux and intentionally pinned.
    (ROOT / 'app/requirements.txt').write_text(header + ''.join(
        f'{name}=={version}\n' for name, version in sorted(versions.items())), encoding='utf-8')
    print(f'Pinned {len(versions)} installed runtime packages.')


if __name__ == '__main__':
    main()
