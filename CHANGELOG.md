## QCUtils Changelog

<!-- version list -->

### v0.8.0 - 2026-04-02

- [Feature] Introduced dataset utilities (`datasets.py`) for common measurement operations
- [Feature] Reworked dataset storage pipeline:
  - Use Zarr for live/in-progress measurements to avoid file locking issues
  - Finalize and export to NetCDF (`.nc`) at measurement completion
  - Added support for partial `.nc` snapshots during sweeps (experimental; may fail on Windows)
- [Feature] Added `update_measurement_path` and improved disk finalization logic for better artifact handling and DB consistency
- [Feature] Increased WebSocket server max message size to 100 MB for large dataset handling
- [Fix] Fixed measurement tabulation issues
- [Fix] Fixed circular sweep behavior
- [Fix] Fixed segmented sweep handling
- [Fix] Fixed MultiChannelParameter handling and related instrument behavior
- [Fix] Fixed measurement bugs (general measurement stability improvements)
- [Fix] Fixed instrument snapshot errors when parameters are `np.ndarray`
- [Fix] Added JSON sanitization for NumPy types in WebSocket responses
- [Misc] Moved `live_db` module to shared location for reuse across projects
- [Misc] Updated dependency versions (including `websockets > 16.0`, zhinst compatibility updates, and numcodecs downgrade)
- [Misc] Internal refactors and consolidation of dataset-related changes
- [Misc] Added and reorganized QCoDeS parameter class support
- [Misc] General cleanup, refactoring, and merge of multiple feature/fix branches (datasets, sweeps, measurement handling)

#### Merge Requests included in this release
- !32 - Log space sweep
- !31 - Station remove parameter
- !30 - Measure tabulate fix
- !29 - Circular sweep fix
- !28 - MultiChannelParameter fix
- !27 - Basel safe sweep (multichannelparameter adjustments)
- !26 - Bux fix measurement
- !25 - Segmented sweep fix
- !24 - QCoDeS parameter classes
- !23 - QCoDeS parameter classes

#### Contributors
- Spandan Anupam
- Simon Schreibing
- Jyotirmaya Shivottam
- Julius
