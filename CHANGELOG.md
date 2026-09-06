## Qanary Changelog

<!-- version list -->

## v0.9.0 (2026-09-06)

### Bug Fixes

- Adapt to qcodes 0.59
  ([`264ea9a`](https://gitlab.com/squad-lab/qcutils/-/commit/264ea9a07f8f800fb12c72b4297802bbcb2c5642))

- Continue sweeps through transient Zarr errors
  ([`6cef1f6`](https://gitlab.com/squad-lab/qcutils/-/commit/6cef1f63767dde077d3a2674ce51405e5b23855b))

- Fixed buffered sweep progress tracking and integration tests
  ([`ba85146`](https://gitlab.com/squad-lab/qcutils/-/commit/ba851460b117179da6e0b825a9d48a80274f9ffb))

- Import GitPython lazily so a git binary is not required
  ([`6491b34`](https://gitlab.com/squad-lab/qcutils/-/commit/6491b345e98fb4f4c6be6339e519cedf291e40cc))

### Build System

- Require Python 3.13 and depend on qimchi-connect
  ([`817b91d`](https://gitlab.com/squad-lab/qcutils/-/commit/817b91d7eb07b1b63615aa4d591f114f64114528))

### Chores

- Store all text files with LF line endings
  ([`d90b6b8`](https://gitlab.com/squad-lab/qcutils/-/commit/d90b6b8ef7fc9de6db49c05f61fa5a332617779f))

- **deps**: Depend on qcdrivers and require qcodes 0.59
  ([`867abe5`](https://gitlab.com/squad-lab/qcutils/-/commit/867abe5fe5b56eefdb3be5cba2155d6b2d05b35f))

### Code Style

- Format examples with ruff
  ([`88191c7`](https://gitlab.com/squad-lab/qcutils/-/commit/88191c7c9365ac3066c3f2b5b5e82a10613e0cef))

- Open multi-line docstrings on their own line
  ([`61b336d`](https://gitlab.com/squad-lab/qcutils/-/commit/61b336d7000d0df500b4814d4f8a504ea0f0e83b))

### Continuous Integration

- Add release, lint, coverage, and Pages pipelines
  ([`990b4ce`](https://gitlab.com/squad-lab/qcutils/-/commit/990b4cedbf0d476af571e0902a1b77ecf374e36b))

- Install portable git for the Windows test job
  ([`388b717`](https://gitlab.com/squad-lab/qcutils/-/commit/388b7174284b58d94fd1af5204889771e0095459))

- Publish to PyPI and drop the git-source workaround
  ([`619701f`](https://gitlab.com/squad-lab/qcutils/-/commit/619701f3bddc9c7b4e70c9f66cd1203e25a4276e))

### Documentation

- Add Sphinx documentation and refresh the README
  ([`9c12c7d`](https://gitlab.com/squad-lab/qcutils/-/commit/9c12c7da4146dddbf5ae823c1ff7d4469dd86beb))

- Document PyPI installation and point buffered nodes at qcdrivers
  ([`2852542`](https://gitlab.com/squad-lab/qcutils/-/commit/28525420272f47eb7ece67e6f69d70a58b3d59a6))

### Features

- Publish live measurements through qimchi-connect
  ([`2d8aaf5`](https://gitlab.com/squad-lab/qcutils/-/commit/2d8aaf5d00f7a57c58e4cc804ffc010eee006fd4))

### Performance Improvements

- Persist only changed Zarr keys at disk checkpoints
  ([`81e4d03`](https://gitlab.com/squad-lab/qcutils/-/commit/81e4d03f485e8be781eae7cb161403bb5b838293))

- Source buffered instrument nodes from qcdrivers
  ([`0655f4b`](https://gitlab.com/squad-lab/qcutils/-/commit/0655f4b890e33910d801f82bae7ce3a6c85ae972))

### Refactoring

- Make internal measurement helpers private
  ([`e1750fb`](https://gitlab.com/squad-lab/qcutils/-/commit/e1750fb6119c7b2c3e5fa82dd5c7066311e55906))

- Remove deprecated instrument classes and related code
  ([`2e8e483`](https://gitlab.com/squad-lab/qcutils/-/commit/2e8e483763e7af060155be06620ab382749901b6))

### Testing

- Cover measurements, persistence, sweeps, and parameters
  ([`df75038`](https://gitlab.com/squad-lab/qcutils/-/commit/df750388c91215ca32a78a0f16082f66970fd8d9))

### Breaking Changes

- Python 3.12 is no longer supported. QCUtils requires 3.13.

- Qcutils.buffered.instruments is removed. Import the node classes from
  qcdrivers.buffered.<manufacturer> instead; QCUtils keeps the buffered sweep orchestration.

- The modules qcutils.live_server and qcutils.shared.live_db are removed. Live publishing goes
  through the qimchi-connect package, which QCUtils now depends on. Measurement scripts are
  unaffected.

- These helpers are renamed with a leading underscore: qcutils.sweep.stepper,
  qcutils.logger.setup_logging, qcutils.measure.{register,get,unregister}_memory_store,
  qcutils.measure.get_live_dataset_info, qcutils.measure.list_live_measurements, and
  qcutils.buffered.sweep.{parse_bufsweep_tree,arm_instruments,fetch_results,
  abort_instruments,fetch_dependents_tree}. The measurement-script API is unchanged.


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
