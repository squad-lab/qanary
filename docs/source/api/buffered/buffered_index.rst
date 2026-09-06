Buffered
========

Instrument-internal fast sweeps, where the hardware acquires a block of points
without a round trip per point.

Qanary does the orchestration -- parsing the node tree, arming the
instruments, fetching results and reporting progress. The instrument-specific
nodes themselves live in `QCDrivers <https://gitlab.com/squad-lab/qcdrivers>`_
under ``qcdrivers.buffered``, grouped by manufacturer.

``qanary.buffered.sweep`` exports no public names: every helper in it is
internal and underscore-prefixed, driven by :func:`qanary.measure.run` rather
than called directly. What follows is the contract between the two packages,
which is what a measurement script and a driver author both need.

The sweep tree
--------------

A buffered measurement is described by a nested dictionary handed to
:func:`qanary.measure.run`. Each level names one instrument node, the sweeps
it drives, and the child nodes triggered beneath it:

.. code-block:: python

    buffered_sweep = {
        "instrument": NodeBaselDAC(inst=dac),
        "sweeps": [gate_sweep],
        "nodes": [
            {
                "instrument": NodeMFLI(inst=lockin),
                "dependent": [lockin_r, lockin_p],
                "grid_mode": "exact",
                "input_trigger": "trigger_in_1",
                "trigger_level": 0.3,
            }
        ],
    }

    measure.run([buffered_sweep], dependents=[], **run_dict)

The root node sweeps; leaf nodes acquire. Qanary walks the tree to size the
dataset, arms every instrument, runs the sweep, then reshapes what comes back
into the dataset's coordinates.

The node contract
-----------------

Any class used as an ``"instrument"`` must provide these, which is all Qanary
calls on it:

.. list-table::
    :header-rows: 1
    :widths: 26 74

    * - Method
      - Purpose
    * - ``register_sweep``
      - Accept one or two sweeps and program the geometry into the hardware.
    * - ``register_dependent``
      - Accept the parameters to acquire and configure the acquisition.
    * - ``run_sweep``
      - Start the programmed sweep.
    * - ``fetch``
      - Return the acquired block, shaped to the registered geometry.
    * - ``abort``
      - Stop a running acquisition; a no-op if nothing is running.
    * - ``toplevel``
      - Whether this node is the root of the tree.

Nodes are typed against a structural ``SweepLike`` protocol rather than
importing :class:`qanary.sweep.Sweep`, so QCDrivers does not depend on
Qanary. :class:`~qanary.sweep.CircularSweep` and
:class:`~qanary.sweep.SegmentedSweep` satisfy it too.

See :doc:`../../tutorials/tutorials_index` for worked examples.
