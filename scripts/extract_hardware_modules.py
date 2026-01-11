#!/usr/bin/env python3
"""
Linux Kernel Hardware Module Extractor

Extracts hardware module information from the Linux kernel source:
- Compatible strings (from device tree bindings)
- Hardware models (from driver data structures)
- Driver file paths (relative)
- First appeared in version (git tags only)

Usage:
    python3 scripts/extract_hardware_modules.py [options]

Options:
    --output-format     Output format: csv, json, or tsv (default: csv)
    --output-file       Output file path (default: stdout)
    --drivers-path      Path to drivers directory (default: drivers/)
    --limit             Limit number of entries to process (for testing)
    --parallel          Number of parallel git operations (default: 4)
    --skip-version      Skip version lookup (faster, for testing)
    --verbose           Enable verbose output
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Iterator


@dataclass
class HardwareModule:
    """Represents a hardware module entry."""
    compatible_string: str
    hardware_model: str
    driver_file: str
    first_version: str
    subsystem: str


class KernelModuleExtractor:
    """Extracts hardware module information from Linux kernel source."""

    # Regex patterns for extracting compatible strings
    COMPATIBLE_PATTERNS = [
        # Standard .compatible = "vendor,device" pattern
        re.compile(
            r'\.compatible\s*=\s*"([^"]+)"',
            re.MULTILINE
        ),
        # OF_DEVICE_COMPAT macro pattern
        re.compile(
            r'OF_DEVICE_COMPAT\s*\(\s*"([^"]+)"\s*\)',
            re.MULTILINE
        ),
    ]

    # Pattern to extract hardware model from .data field
    DATA_PATTERN = re.compile(
        r'\.compatible\s*=\s*"([^"]+)"\s*,\s*\.data\s*=\s*&?(\w+)',
        re.MULTILINE | re.DOTALL
    )

    # Pattern to find of_device_id table definitions
    OF_DEVICE_ID_PATTERN = re.compile(
        r'static\s+const\s+struct\s+of_device_id\s+(\w+)\s*\[\s*\]\s*=\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}',
        re.MULTILINE | re.DOTALL
    )

    def __init__(self, kernel_root: str, parallel: int = 4, verbose: bool = False):
        self.kernel_root = Path(kernel_root)
        self.parallel = parallel
        self.verbose = verbose
        self._version_cache: Dict[str, str] = {}
        self._tag_list: Optional[List[str]] = None
        self._is_shallow: Optional[bool] = None

    def is_shallow_clone(self) -> bool:
        """Check if this is a shallow git clone."""
        if self._is_shallow is not None:
            return self._is_shallow

        shallow_file = self.kernel_root / '.git' / 'shallow'
        if shallow_file.exists():
            self._is_shallow = True
            return True

        # Also check commit count - full kernel has millions of commits
        try:
            result = subprocess.run(
                ['git', 'rev-list', '--count', 'HEAD'],
                cwd=self.kernel_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                count = int(result.stdout.strip())
                # If less than 100k commits, likely shallow
                self._is_shallow = count < 100000
                return self._is_shallow
        except:
            pass

        self._is_shallow = False
        return False

    def log(self, message: str) -> None:
        """Print verbose log messages."""
        if self.verbose:
            print(f"[INFO] {message}", file=sys.stderr)

    def get_sorted_tags(self) -> List[str]:
        """Get sorted list of kernel version tags."""
        if self._tag_list is not None:
            return self._tag_list

        self.log("Fetching and sorting git tags...")
        try:
            result = subprocess.run(
                ['git', 'tag', '-l', 'v*'],
                cwd=self.kernel_root,
                capture_output=True,
                text=True,
                check=True
            )
            tags = result.stdout.strip().split('\n')

            # Filter to only include release tags (v X.Y or vX.Y.Z format)
            version_pattern = re.compile(r'^v(\d+)\.(\d+)(\.(\d+))?(-rc(\d+))?$')
            valid_tags = []
            for tag in tags:
                match = version_pattern.match(tag)
                if match:
                    major = int(match.group(1))
                    minor = int(match.group(2))
                    patch = int(match.group(4)) if match.group(4) else 0
                    rc = int(match.group(6)) if match.group(6) else 999  # Non-rc is higher
                    valid_tags.append((tag, (major, minor, patch, rc)))

            # Sort by version tuple
            valid_tags.sort(key=lambda x: x[1])
            self._tag_list = [tag for tag, _ in valid_tags]

            self.log(f"Found {len(self._tag_list)} valid tags")
            return self._tag_list

        except subprocess.CalledProcessError as e:
            self.log(f"Error fetching tags: {e}")
            return []

    def find_first_version(self, file_path: str, compatible_string: str) -> str:
        """Find the first git tag where a compatible string appeared."""
        cache_key = f"{file_path}:{compatible_string}"
        if cache_key in self._version_cache:
            return self._version_cache[cache_key]

        try:
            # First, check if file exists in current tree
            full_path = self.kernel_root / file_path
            if not full_path.exists():
                return "unknown"

            # Use git log to find when the compatible string was first added
            # Search for the compatible string in the file's history
            escaped_compat = re.escape(compatible_string)

            result = subprocess.run(
                [
                    'git', 'log', '--oneline', '--all', '--follow',
                    '-S', compatible_string,
                    '--', file_path
                ],
                cwd=self.kernel_root,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0 or not result.stdout.strip():
                # Try alternative: search in blame
                self._version_cache[cache_key] = "unknown"
                return "unknown"

            # Get the oldest commit that contains this string
            commits = result.stdout.strip().split('\n')
            if not commits:
                self._version_cache[cache_key] = "unknown"
                return "unknown"

            oldest_commit = commits[-1].split()[0]  # Last commit is oldest

            # Find the first tag that contains this commit
            result = subprocess.run(
                ['git', 'describe', '--tags', '--contains', oldest_commit],
                cwd=self.kernel_root,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0 and result.stdout.strip():
                # Output is like "v5.10~123^2~45" - extract base tag
                tag_output = result.stdout.strip()
                # Get just the tag part (before ~ or ^)
                tag = re.split(r'[~^]', tag_output)[0]
                self._version_cache[cache_key] = tag
                return tag

            # Alternative: use git tag --contains
            result = subprocess.run(
                ['git', 'tag', '--contains', oldest_commit, '--sort=v:refname'],
                cwd=self.kernel_root,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0 and result.stdout.strip():
                tags = result.stdout.strip().split('\n')
                # Filter for version tags and get the oldest
                version_pattern = re.compile(r'^v\d+\.\d+')
                version_tags = [t for t in tags if version_pattern.match(t)]
                if version_tags:
                    self._version_cache[cache_key] = version_tags[0]
                    return version_tags[0]

            self._version_cache[cache_key] = "unknown"
            return "unknown"

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            self.log(f"Error finding version for {compatible_string}: {e}")
            self._version_cache[cache_key] = "error"
            return "error"

    def extract_subsystem(self, file_path: str) -> str:
        """Extract subsystem name from driver file path."""
        parts = Path(file_path).parts
        if len(parts) >= 2 and parts[0] == 'drivers':
            return parts[1]
        elif len(parts) >= 1:
            return parts[0]
        return "unknown"

    def extract_from_file(self, file_path: Path) -> Iterator[tuple]:
        """Extract compatible strings and hardware models from a driver file."""
        try:
            content = file_path.read_text(errors='ignore')
        except Exception as e:
            self.log(f"Error reading {file_path}: {e}")
            return

        rel_path = str(file_path.relative_to(self.kernel_root))

        # Find all of_device_id table definitions
        for match in self.OF_DEVICE_ID_PATTERN.finditer(content):
            table_name = match.group(1)
            table_content = match.group(2)

            # Extract compatible strings and data from this table
            for data_match in self.DATA_PATTERN.finditer(table_content):
                compatible = data_match.group(1)
                hw_model = data_match.group(2)
                yield (compatible, hw_model, rel_path)

            # Collect compatible strings that already have .data field
            data_compatibles = set(m.group(1) for m in self.DATA_PATTERN.finditer(table_content))

            # Also extract compatible strings without .data field
            for compat_match in self.COMPATIBLE_PATTERNS[0].finditer(table_content):
                compatible = compat_match.group(1)
                # Check if we already found this with a data field
                if compatible not in data_compatibles:
                    # Try to find hardware model from surrounding context
                    hw_model = self._guess_hw_model(compatible, table_name, content)
                    yield (compatible, hw_model, rel_path)

    def _guess_hw_model(self, compatible: str, table_name: str, content: str) -> str:
        """Try to guess hardware model from compatible string or context."""
        # Extract model from compatible string format: "vendor,model"
        if ',' in compatible:
            parts = compatible.split(',', 1)
            if len(parts) == 2:
                return parts[1]  # Return device part
        return compatible

    def find_driver_files(self, drivers_path: str = "drivers") -> Iterator[Path]:
        """Find all C source files that might contain of_device_id tables."""
        search_paths = [
            self.kernel_root / drivers_path,
            self.kernel_root / "sound",
            self.kernel_root / "net",
        ]

        for search_path in search_paths:
            if not search_path.exists():
                continue

            for file_path in search_path.rglob("*.c"):
                yield file_path

    def extract_all(self, drivers_path: str = "drivers",
                   limit: Optional[int] = None,
                   skip_version: bool = False) -> List[HardwareModule]:
        """Extract all hardware modules from kernel source."""
        modules = []
        seen_compatibles = set()
        count = 0

        # Check for shallow clone
        if not skip_version and self.is_shallow_clone():
            self.log("WARNING: Detected shallow git clone - version lookup disabled")
            self.log("For version information, use a full clone with tags:")
            self.log("  git fetch --unshallow && git fetch --tags")
            skip_version = True

        # Check for tags
        if not skip_version:
            tags = self.get_sorted_tags()
            if not tags:
                self.log("WARNING: No git tags found - version lookup disabled")
                self.log("Fetch tags with: git fetch --tags")
                skip_version = True

        self.log("Scanning driver files for compatible strings...")

        # First pass: extract all compatible strings
        entries = []
        for file_path in self.find_driver_files(drivers_path):
            for compatible, hw_model, rel_path in self.extract_from_file(file_path):
                if compatible not in seen_compatibles:
                    seen_compatibles.add(compatible)
                    entries.append((compatible, hw_model, rel_path))
                    count += 1
                    if limit and count >= limit:
                        break
            if limit and count >= limit:
                break

        self.log(f"Found {len(entries)} unique compatible strings")

        # Second pass: find versions (optionally in parallel)
        if skip_version:
            for compatible, hw_model, rel_path in entries:
                subsystem = self.extract_subsystem(rel_path)
                modules.append(HardwareModule(
                    compatible_string=compatible,
                    hardware_model=hw_model,
                    driver_file=rel_path,
                    first_version="n/a",
                    subsystem=subsystem
                ))
        else:
            self.log("Looking up first version for each compatible string...")

            def process_entry(entry):
                compatible, hw_model, rel_path = entry
                version = self.find_first_version(rel_path, compatible)
                subsystem = self.extract_subsystem(rel_path)
                return HardwareModule(
                    compatible_string=compatible,
                    hardware_model=hw_model,
                    driver_file=rel_path,
                    first_version=version,
                    subsystem=subsystem
                )

            with ThreadPoolExecutor(max_workers=self.parallel) as executor:
                futures = {executor.submit(process_entry, entry): entry
                          for entry in entries}

                for i, future in enumerate(as_completed(futures)):
                    try:
                        module = future.result()
                        modules.append(module)
                        if (i + 1) % 100 == 0:
                            self.log(f"Processed {i + 1}/{len(entries)} entries")
                    except Exception as e:
                        entry = futures[future]
                        self.log(f"Error processing {entry[0]}: {e}")

        # Sort by subsystem, then by compatible string
        modules.sort(key=lambda m: (m.subsystem, m.compatible_string))

        return modules


def output_csv(modules: List[HardwareModule], output_file) -> None:
    """Output modules in CSV format."""
    writer = csv.writer(output_file)
    writer.writerow(['compatible_string', 'hardware_model', 'driver_file',
                    'first_version', 'subsystem'])
    for module in modules:
        writer.writerow([
            module.compatible_string,
            module.hardware_model,
            module.driver_file,
            module.first_version,
            module.subsystem
        ])


def output_tsv(modules: List[HardwareModule], output_file) -> None:
    """Output modules in TSV format."""
    writer = csv.writer(output_file, delimiter='\t')
    writer.writerow(['compatible_string', 'hardware_model', 'driver_file',
                    'first_version', 'subsystem'])
    for module in modules:
        writer.writerow([
            module.compatible_string,
            module.hardware_model,
            module.driver_file,
            module.first_version,
            module.subsystem
        ])


def output_json(modules: List[HardwareModule], output_file) -> None:
    """Output modules in JSON format."""
    data = [asdict(m) for m in modules]
    json.dump(data, output_file, indent=2)
    output_file.write('\n')


def main():
    parser = argparse.ArgumentParser(
        description='Extract hardware module information from Linux kernel source',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--output-format', '-f',
        choices=['csv', 'json', 'tsv'],
        default='csv',
        help='Output format (default: csv)'
    )
    parser.add_argument(
        '--output-file', '-o',
        type=str,
        default=None,
        help='Output file path (default: stdout)'
    )
    parser.add_argument(
        '--kernel-root', '-k',
        type=str,
        default='.',
        help='Path to kernel source root (default: current directory)'
    )
    parser.add_argument(
        '--drivers-path', '-d',
        type=str,
        default='drivers',
        help='Path to drivers directory relative to kernel root (default: drivers)'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Limit number of entries to process (for testing)'
    )
    parser.add_argument(
        '--parallel', '-p',
        type=int,
        default=4,
        help='Number of parallel git operations (default: 4)'
    )
    parser.add_argument(
        '--skip-version', '-s',
        action='store_true',
        help='Skip version lookup (faster, for testing)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Verify kernel root
    kernel_root = Path(args.kernel_root).resolve()
    if not (kernel_root / 'Kconfig').exists():
        print(f"Error: {kernel_root} does not appear to be a kernel source tree",
              file=sys.stderr)
        sys.exit(1)

    # Create extractor and run
    extractor = KernelModuleExtractor(
        kernel_root=str(kernel_root),
        parallel=args.parallel,
        verbose=args.verbose
    )

    modules = extractor.extract_all(
        drivers_path=args.drivers_path,
        limit=args.limit,
        skip_version=args.skip_version
    )

    # Output results
    if args.output_file:
        output_path = Path(args.output_file)
        with open(output_path, 'w', newline='') as f:
            if args.output_format == 'csv':
                output_csv(modules, f)
            elif args.output_format == 'tsv':
                output_tsv(modules, f)
            elif args.output_format == 'json':
                output_json(modules, f)
        if args.verbose:
            print(f"Wrote {len(modules)} entries to {output_path}", file=sys.stderr)
    else:
        if args.output_format == 'csv':
            output_csv(modules, sys.stdout)
        elif args.output_format == 'tsv':
            output_tsv(modules, sys.stdout)
        elif args.output_format == 'json':
            output_json(modules, sys.stdout)

    if args.verbose:
        print(f"\nSummary:", file=sys.stderr)
        print(f"  Total modules: {len(modules)}", file=sys.stderr)
        subsystems = {}
        for m in modules:
            subsystems[m.subsystem] = subsystems.get(m.subsystem, 0) + 1
        print(f"  Subsystems:", file=sys.stderr)
        for sub, count in sorted(subsystems.items(), key=lambda x: -x[1])[:10]:
            print(f"    {sub}: {count}", file=sys.stderr)


if __name__ == '__main__':
    main()
