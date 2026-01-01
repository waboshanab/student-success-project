Data directory layout:

- `data/raw/` - raw ingested files (keep immutable copies)
  - `oulad/` - Open University Learning Analytics dataset files
  - `uci/` - UCI Student Performance dataset files
- `data/interim/` - intermediate artifacts during processing
- `data/processed/` - final processed feature tables and model inputs

Place dataset READMEs or manifests inside corresponding subfolders.