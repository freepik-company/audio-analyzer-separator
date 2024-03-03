import sys
import subprocess
import torch

from pathlib import Path
from typing import List, Union


def demix(paths: List[Path], demix_dir: Path, device: Union[str, torch.device]):
  """Demixes the audio file into its sources."""
  todos = []
  demix_paths = []
  for path in paths:
    out_dir = demix_dir / 'htdemucs' / path.stem
    demix_paths.append(out_dir)
    if out_dir.is_dir():
      if (
        (out_dir / 'bass.wav').is_file() and
        (out_dir / 'drums.wav').is_file() and
        (out_dir / 'other.wav').is_file() and
        (out_dir / 'vocals.wav').is_file()
      ):
        continue
    todos.append(path)

  existing = len(paths) - len(todos)
  print(f'=> Found {existing} tracks already demixed, {len(todos)} to demix.')

  static_models_dir = demix_dir.parent / 'static_models'
  
  # Convert to absolute path to avoid any relative path issues
  absolute_static_models_dir = static_models_dir.resolve()

  if todos:
    subprocess.run(
      [
        sys.executable, '-m', 'demucs.separate',
        '--out', demix_dir.as_posix(),
        '--name', 'htdemucs',
        '--device', str(device),
        # one folder up inside static_models
        "--repo", absolute_static_models_dir.as_posix(),
        "-n", "htdemucs", # ADDED PRECISA ter o YAML senão  n funciona: https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json
        *[path.as_posix() for path in todos],
      ],
      check=True,
    )

  return demix_paths
