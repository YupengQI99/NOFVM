"""Host resource sampling and guards used by the long-running experiment scripts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[2]


def _save(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temp.replace(path)


def _cpu_temperatures():
    cpu = []
    for p in Path('/sys/class/hwmon').glob('hwmon*'):
        name = p/'name'
        if name.exists() and name.read_text().strip() in ('k10temp', 'coretemp'):
            cpu += [float(t.read_text())/1000 for t in p.glob('temp*_input')]
    return cpu


def resources():
    """CPU temperature, process RSS, available memory, free disk and load average."""
    mem = {line.split(':')[0]: float(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    status = Path('/proc/self/status').read_text().splitlines()
    rss = float(next(line.split()[1] for line in status if line.startswith('VmRSS:')))/2**20
    cpu = _cpu_temperatures()
    return dict(time=time.time(), cpu_max_c=max(cpu) if cpu else None, rss_gib=rss,
                available_gib=mem['MemAvailable']/2**20, disk_free_gib=shutil.disk_usage(ROOT).free/2**30,
                load_average=list(os.getloadavg()))


class Monitor:
    """Sample resources every five seconds into ``resource_samples.json``.

    The process exits with status 90, after writing ``resource_stop.json``, when the CPU
    temperature reaches ``max_cpu_c``, the process RSS exceeds ``max_rss_gib``, available
    memory falls below ``min_available_gib`` or free disk falls below ``min_disk_gib``.
    A limit set to None is not checked.
    """
    def __init__(self, output, max_cpu_c=90, max_rss_gib=40, min_available_gib=8, min_disk_gib=4):
        self.output = output; self.stop = threading.Event(); self.records = []
        self.limits = dict(max_cpu_c=max_cpu_c, max_rss_gib=max_rss_gib,
                           min_available_gib=min_available_gib, min_disk_gib=min_disk_gib)
        self.thread = threading.Thread(target=self.run, daemon=True)

    def exceeded(self, row):
        limit = self.limits
        return ((limit['max_cpu_c'] is not None and row['cpu_max_c'] is not None and row['cpu_max_c'] >= limit['max_cpu_c'])
                or (limit['max_rss_gib'] is not None and row['rss_gib'] > limit['max_rss_gib'])
                or (limit['min_available_gib'] is not None and row['available_gib'] < limit['min_available_gib'])
                or (limit['min_disk_gib'] is not None and row['disk_free_gib'] < limit['min_disk_gib']))

    def run(self):
        while not self.stop.is_set():
            row = resources(); self.records.append(row)
            _save(self.output/'resource_samples.json', self.records)
            if self.exceeded(row):
                _save(self.output/'resource_stop.json', row); os._exit(90)
            self.stop.wait(5)

    def __enter__(self):
        self.thread.start(); return self

    def __exit__(self, *args):
        self.stop.set(); self.thread.join(timeout=10)


def guard(output):
    """Return CPU/GPU temperatures and free disk; raise if either temperature reaches 90 C or disk < 4 GiB."""
    cpu = []
    for directory in Path("/sys/class/hwmon").glob("hwmon*"):
        namepath = directory / "name"
        if namepath.exists() and namepath.read_text().strip() in ("k10temp", "coretemp"):
            for path in directory.glob("temp*_input"):
                cpu.append(float(path.read_text().strip())/1000)
    gpu = float(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
        text=True, timeout=15).strip().splitlines()[0])
    free = shutil.disk_usage(output).free/2**30
    record = dict(cpu_max_c=max(cpu) if cpu else None, gpu_c=gpu, free_gib=free)
    if gpu >= 90 or any(t >= 90 for t in cpu) or free < 4:
        raise RuntimeError(f"Resource guard stopped this experiment: {record}")
    return record
