Buffered
========

Instrument-internal fast sweeps, where the hardware acquires a block of points
without a round trip per point.

QCUtils does the orchestration -- parsing the node tree, arming the
instruments, fetching results and reporting progress. The instrument-specific
nodes themselves live in `QCDrivers <https://gitlab.com/squad-lab/qcdrivers>`_
under ``qcdrivers.buffered``, grouped by manufacturer.

.. automodule:: qcutils.buffered.sweep
    :members:
    :undoc-members:
    :show-inheritance:
