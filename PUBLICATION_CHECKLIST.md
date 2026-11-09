# GitHub public-release checklist

- [ ] Verify all original source files for passwords, API keys, personal data.
- [ ] Choose an explicit open-source license if desired.
- [ ] Verify rights to redistribute SpeechOcean762-derived labels/transcripts.
- [ ] Verify model weights/data are not bundled (only source and derived tables).
- [ ] Verify `MANIFEST_SHA256.csv` with `python scripts/verify_package.py`.
- [ ] Confirm paths and dependency versions on your Kaggle runtime.
- [ ] Decide which raw derived CSVs can be shared openly; restrict if needed.
- [ ] Add article authors and title to `CITATION.cff` only when confirmed.
- [ ] Update the manuscript Code Availability statement with a live, accessible
      repository URL only after publishing and testing access.
- [ ] Consider archiving a final code release using Zenodo with a DOI.
