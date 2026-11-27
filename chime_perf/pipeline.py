"""Configure EOPL execution and image I/O for CHIME's spatial axes."""

import inspect
import time
from numbers import Integral
from pathlib import Path

import dask
import xarray as xr
from eopl import Pipeline, image_t
from eopl.data import require_cupy

from chime_perf.algorithms.backends import native_module, validate_backend


class ChimePipeline(Pipeline):
    def set_run_options(self, *, backend="python", dask_chunks=None, **options):
        self.backend = validate_backend(backend)
        if self.backend == "cpp":
            threads = options.get("num_cores", 1)
            if isinstance(threads, bool) or not isinstance(threads, Integral) or threads < 1:
                raise ValueError("num_cores must be a positive integer for OpenMP.")
            native_module()
        if dask_chunks is None:
            dask_chunks = {"ACT": 128, "ALT": 128}
        if (
            not isinstance(dask_chunks, dict)
            or not dask_chunks
            or any(
                axis not in ("ACT", "ALT") or type(size) is not int or size < 1
                for axis, size in dask_chunks.items()
            )
        ):
            raise ValueError("dask_chunks must map ACT or ALT to positive integers.")
        self.dask_chunks = dask_chunks.copy()
        self.pixel_chunks = {"ACT": -1, "ALT": -1, "SPE": -1, **self.dask_chunks}

        if options.pop("use_cuda", False) and self.backend != "cuda":
            raise ValueError("Select backend: cuda to enable GPU execution.")
        if self.backend == "cuda":
            require_cupy()
        super().set_run_options(use_cuda=self.backend == "cuda", **options)
        self.manifest["processing_backend"] = self.backend
        if self.backend == "dask":
            self.manifest["dask_chunks"] = self.dask_chunks.copy()

    def open_product(self, path):
        """Open pixel cubes in spatial chunks, keeping each spectrum together."""
        chunks = self.pixel_chunks if self.backend == "dask" else None
        return xr.open_dataset(path, chunks=chunks)

    def prepare_image(self, data):
        if self.backend == "dask":
            image = data.image()
            if not data.is_dask:
                chunks = {
                    axis: size
                    for axis, size in self.pixel_chunks.items()
                    if axis in image.dims
                }
                return image_t(image.chunk(chunks))
            if "SPE" in image.dims and len(image.chunks[image.get_axis_num("SPE")]) > 1:
                return image_t(image.chunk({"SPE": -1}))
        if self.backend != "dask" and data.is_dask:
            return image_t(data.image().compute())
        return data

    def read_input(self, path):
        if self.backend != "dask" or self._is_skipping:
            return super().read_input(str(path), dask=False)

        self.manifest["input"] = {"file": str(Path(path).resolve())}
        with self.open_product(path) as dataset:
            data = self._restore_dataset(dataset)
        if isinstance(data, tuple):
            raise ValueError("Select one image variable from the input product.")
        self._last_bkp_time = time.time()
        return data

    def read_radiance(self, path):
        """Read the image separately because the saved coefficients use another grid."""
        with self.open_product(path) as source:
            image = source["radiance"]
            if self.backend != "dask":
                image = image.load()
        self.manifest["input"] = {"file": str(Path(path).resolve())}
        return image_t(image)

    def run_operation(self, data, func, args=None):
        args = dict(args or {})
        parameters = inspect.signature(func).parameters
        if "backend" in parameters:
            args["backend"] = self.backend
        if "threads" in parameters:
            args["threads"] = self.run_options.num_cores
        return super().run_operation(
            data,
            func,
            args,
            dask_acc_chunks=["ACT", "ALT", "SPE", "material", "detector_column", "sample_SPE"],
        )

    def set_dask(self, data, dask=False, dask_chunk_direction="ACT"):
        if dask and self.backend == "dask":
            # EOPL calls this after a checkpoint. Preserve image chunks and keep
            # small eager coefficient arrays on the CPU.
            return data
        return super().set_dask(data, dask, dask_chunk_direction)

    def save_bkp(self, data, label):
        with dask.config.set(num_workers=self.run_options.num_cores):
            return super().save_bkp(
                data, label, dask=self.backend == "dask", dask_chunk_direction="ACT"
            )

    def save_output(self, data, label):
        with dask.config.set(num_workers=self.run_options.num_cores):
            return super().save_output(data, label)
