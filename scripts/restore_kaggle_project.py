#!/usr/bin/env python3
"""Restore archived Kaggle project/ content without replacing existing frozen results.

Usage in Kaggle:
  python scripts/restore_kaggle_project.py --target /kaggle/working/acoustic_linguistic_research
  # Add --overwrite only after making a separate backup.

Does not include raw WAV/audio or the original Parquet file; obtain them from the
public dataset separately, respecting the license and original split.
"""
import argparse
from pathlib import Path
import zipfile

repo=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path,
         default=repo/'archive/source_checkpoints/asr_fluency_step01-07_complete.zip')
    parser.add_argument('--target',type=Path,default=Path('/kaggle/working/acoustic_linguistic_research'))
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    if not args.archive.exists():raise FileNotFoundError(args.archive)
    count=0; skipped=0
    with zipfile.ZipFile(args.archive) as z:
        for info in z.infolist():
            if not info.filename.startswith('project/') or info.is_dir():continue
            remainder=Path(info.filename).relative_to('project')
            if '..' in remainder.parts:raise ValueError('Unsafe archive member')
            dest=args.target/remainder
            if dest.exists() and not args.overwrite:
                skipped+=1;continue
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(z.read(info.filename));count+=1
    print('Restored:',count,'Skipped existing:',skipped,'Target:',args.target)
    print('Original Parquet and audio are NOT included in this archive.')
if __name__=='__main__':main()
