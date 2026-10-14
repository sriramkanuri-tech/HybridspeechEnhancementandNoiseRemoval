"""Audio device helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import sounddevice as sd


@dataclass
class DeviceInfo:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    hostapi: str


def query_devices() -> List[DeviceInfo]:
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    out = []
    for i, d in enumerate(devices):
        ha = hostapis[d["hostapi"]]["name"] if d["hostapi"] < len(hostapis) else ""
        out.append(DeviceInfo(
            index=i,
            name=d["name"],
            max_input_channels=int(d["max_input_channels"]),
            max_output_channels=int(d["max_output_channels"]),
            default_samplerate=float(d["default_samplerate"]),
            hostapi=ha,
        ))
    return out


def input_devices() -> List[DeviceInfo]:
    return [d for d in query_devices() if d.max_input_channels > 0]


def output_devices() -> List[DeviceInfo]:
    return [d for d in query_devices() if d.max_output_channels > 0]


def is_virtual_name(name: str) -> bool:
    name_l = name.lower()
    markers = ("cable", "voicemeeter", "virtual", "vb-audio", "vb audio", "wave link", "blackhole")
    return any(m in name_l for m in markers)


def find_virtual_outputs() -> List[DeviceInfo]:
    return [d for d in output_devices() if is_virtual_name(d.name)]


def virtual_output_devices() -> List[DeviceInfo]:
    return [d for d in output_devices() if is_virtual_name(d.name)]


def physical_output_devices() -> List[DeviceInfo]:
    return [d for d in output_devices() if not is_virtual_name(d.name)]


def virtual_input_devices() -> List[DeviceInfo]:
    return [d for d in input_devices() if is_virtual_name(d.name)]


def physical_input_devices() -> List[DeviceInfo]:
    return [d for d in input_devices() if not is_virtual_name(d.name)]
