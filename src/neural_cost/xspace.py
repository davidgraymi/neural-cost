"""Device busy time from XLA/TensorFlow profiler traces (XSpace protobufs).

``jax.profiler`` and ``tf.profiler`` both write an ``*.xplane.pb`` file: a
set of planes (one per host or device), each holding lines (threads, GPU
streams, TPU op timelines) of timed events.  This module reads one with
whichever parser is installed and reports how long each accelerator was
busy, without depending on either framework.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Iterator
from dataclasses import dataclass

#: An event as (plane name, line name, start ns, duration ns).
TraceEvent = tuple[str, str, float, float]


@dataclass(frozen=True, slots=True)
class DeviceTime:
    """Busy time of the accelerators in one profiler trace."""

    #: Seconds each device plane had at least one kernel running, keyed by plane name.
    busy_seconds: dict[str, float]
    #: Kernel or op events counted.
    event_count: int

    @property
    def total_seconds(self) -> float:
        """Busy time summed over devices (device-seconds)."""
        return sum(self.busy_seconds.values())


def device_time(events: Iterator[TraceEvent] | list[TraceEvent]) -> DeviceTime | None:
    """Busy time of every accelerator plane in ``events``.

    Only kernel timelines count: GPU ``Stream`` lines, or the ``XLA Ops``
    line on devices without streams (TPU).  Derived lines such as
    ``XLA Modules`` span the gaps between kernels and are skipped.  Events
    on concurrent streams are merged, so overlapping kernels count once.
    Returns ``None`` when the trace has no accelerator planes (CPU runs).
    """
    by_line: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for plane, line, start, duration in events:
        if not plane.startswith("/device:") or duration <= 0:
            continue
        by_line.setdefault(plane, {}).setdefault(line, []).append((start, start + duration))
    busy: dict[str, float] = {}
    count = 0
    for plane, lines in by_line.items():
        streams = [spans for name, spans in lines.items() if name.startswith("Stream")]
        if not streams:
            streams = [spans for name, spans in lines.items() if name == "XLA Ops"]
        if not streams:
            continue
        intervals = sorted(span for spans in streams for span in spans)
        count += len(intervals)
        busy[plane] = _union_length(intervals) / 1e9
    if not busy:
        return None
    return DeviceTime(busy, count)


def _union_length(intervals: list[tuple[float, float]]) -> float:
    """Total length covered by sorted ``(start, end)`` intervals."""
    total = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    return total + current_end - current_start


def read_events(serialized: bytes, parser: str = "auto") -> list[TraceEvent]:
    """Events from a serialized XSpace.

    ``parser`` is ``"jax"`` (``jax.profiler.ProfileData``), ``"tensorflow"``
    (TensorFlow's generated protobuf) or ``"auto"``: JAX when it is already
    imported, otherwise TensorFlow, so reading a trace never imports a large
    framework the caller is not using.
    """
    import sys

    use_jax = parser == "jax" or (parser == "auto" and "jax" in sys.modules)
    proto = None if use_jax else _xplane_proto()
    if proto is not None:
        space = proto.XSpace()
        space.ParseFromString(serialized)
        events: list[TraceEvent] = []
        for plane in space.planes:
            for line in plane.lines:
                base_ns = line.timestamp_ns
                for event in line.events:
                    events.append(
                        (
                            plane.name,
                            line.name,
                            base_ns + event.offset_ps / 1000,
                            event.duration_ps / 1000,
                        )
                    )
        return events
    import jax

    data = jax.profiler.ProfileData.from_serialized_xspace(serialized)
    return [
        (plane.name, line.name, event.start_ns, event.duration_ns)
        for plane in data.planes
        for line in plane.lines
        for event in line.events
    ]


def read_trace_dir(log_dir: str, parser: str = "auto") -> DeviceTime | None:
    """Device time from the newest ``*.xplane.pb`` written under ``log_dir``."""
    files = glob.glob(os.path.join(log_dir, "**", "*.xplane.pb"), recursive=True)
    if not files:
        return None
    newest = max(files, key=os.path.getmtime)
    with open(newest, "rb") as handle:
        return device_time(read_events(handle.read(), parser))


def _xplane_proto():
    """TensorFlow's generated XSpace protobuf module, if TensorFlow is installed."""
    for module in (
        "tensorflow.tsl.profiler.protobuf.xplane_pb2",
        "tsl.profiler.protobuf.xplane_pb2",
    ):
        try:
            return __import__(module, fromlist=["XSpace"])
        except ImportError:
            continue
    return None
