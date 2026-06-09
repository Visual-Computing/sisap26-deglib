"""
runner.py — Task1Runner: Docker container management for deglib_evp_task1.

Responsibilities
----------------
1. Download the SISAP 2026 dataset via HuggingFace Hub (cached locally).
2. Build the Docker image from the project Dockerfile (optional).
3. Start the container with SISAP-compliant resource limits and volume mounts.
4. Stream container logs in real time, feeding each line to Task1LogParser.
5. Return a structured Task1Result when the container exits.

Resource limits (fixed, matching official SISAP 2026 evaluation):
    --cpus=8
    --memory=24g
    --memory-swap=24g   (no swap beyond the RAM limit)
    --memory-swappiness=0

Volume mounts:
    <hf_cache_dir> → /data:ro   (dataset, read-only)
    ./results      → /results:rw (outputs, read-write)
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

import docker
import docker.errors
from huggingface_hub import hf_hub_download

from .log_parser import Task1LogParser
from .result import Task1Result

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

REPO_ID = "SISAP-Challenges/SISAP2026"
REPO_TYPE = "dataset"

DATASET_FILES: dict[str, str] = {
    "small": "benchmark-dev-wikipedia-bge-m3-small.h5",
    "large": "benchmark-dev-wikipedia-bge-m3.h5",
}

# ---------------------------------------------------------------------------
# SISAP resource limits (must not be changed for official evaluation)
# ---------------------------------------------------------------------------

_NANO_CPUS: int = 8 * 1_000_000_000   # --cpus=8
_MEM_LIMIT: int = 24 * 1024**3         # --memory=24g
_MEM_SWAP: int  = 24 * 1024**3         # --memory-swap=24g  (= RAM limit → no swap)
_SWAPPINESS: int = 0                    # --memory-swappiness=0

# Default Docker image tag
_DEFAULT_IMAGE = "sisap26-deglib-cpp"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class Task1Runner:
    """
    High-level interface for running deglib_evp_task1 inside Docker.

    Parameters
    ----------
    image_tag:
        Name (and optional tag) of the Docker image to use.
    dockerfile_dir:
        Directory containing the Dockerfile. Defaults to the parent of this
        file's package directory (i.e. the project root).
    results_dir:
        Local directory mounted as /results inside the container. Created
        automatically if it does not exist. Defaults to ``<project_root>/results``.
    echo_logs:
        When True (default), every container log line is printed to stdout
        in real time so you can watch progress interactively.
    """

    def __init__(
        self,
        image_tag: str = _DEFAULT_IMAGE,
        dockerfile_dir: Path | None = None,
        results_dir: Path | None = None,
        echo_logs: bool = True,
    ) -> None:
        self.image_tag = image_tag

        # Resolve project root relative to this file: docker_runner/ → project/
        _pkg_dir = Path(__file__).parent
        _project_root = _pkg_dir.parent

        self.dockerfile_dir = Path(dockerfile_dir) if dockerfile_dir else _project_root
        self.results_dir = Path(results_dir) if results_dir else _project_root / "results"
        self.echo_logs = echo_logs

        # Lazy-initialised Docker client
        self._client: docker.DockerClient | None = None

    # ------------------------------------------------------------------
    # Docker client (lazy)
    # ------------------------------------------------------------------

    @property
    def client(self) -> docker.DockerClient:
        if self._client is None:
            try:
                self._client = docker.from_env()
            except docker.errors.DockerException as exc:
                raise RuntimeError(
                    "Cannot connect to the Docker daemon. "
                    "Make sure Docker Desktop (or the Docker daemon) is running."
                ) from exc
        return self._client

    # ------------------------------------------------------------------
    # Dataset management
    # ------------------------------------------------------------------

    def get_dataset_path(self, size: str = "small") -> Path:
        """
        Return the local path to the cached HDF5 dataset file.

        Downloads from HuggingFace on first call; subsequent calls are instant
        because hf_hub_download checks the local cache first.

        Parameters
        ----------
        size:
            ``"small"`` (200 K vectors) or ``"large"`` (6.35 M vectors).
        """
        if size not in DATASET_FILES:
            raise ValueError(
                f"Unknown dataset size {size!r}. Choose from: {list(DATASET_FILES)}"
            )
        filename = DATASET_FILES[size]
        
        # Check local HF cache folder dynamically (safe, generic, and offline-friendly):
        try:
            hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--sisap-challenges--SISAP2026"
            if hf_cache_dir.exists():
                for p in hf_cache_dir.rglob(filename):
                    if p.is_file() and ".no_exist" not in p.parts:
                        print(f"[Task1Runner] Found local cached dataset file at: {p}", flush=True)
                        return p
        except Exception:
            pass

        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type=REPO_TYPE,
            )
        except Exception:
            print("[Task1Runner] HF Hub download failed. Falling back to local cache...", flush=True)
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type=REPO_TYPE,
                local_files_only=True,
            )
        return Path(local_path)

    def get_data_dir(self, size: str = "small") -> Path:
        """
        Return the directory of the cached HDF5 file (used as the volume-mount
        source for ``/data:ro`` inside the container).
        """
        return self.get_dataset_path(size).parent

    # ------------------------------------------------------------------
    # Image management
    # ------------------------------------------------------------------

    def build_image(self, tag: str | None = None, force: bool = False) -> None:
        """
        Build the Docker image from the project Dockerfile.

        If the image already exists and ``force`` is ``False``, the build is
        skipped — this makes it safe to call before every run.

        Parameters
        ----------
        tag:
            Override the image tag. Defaults to ``self.image_tag``.
        force:
            When ``True``, force a fresh build with all caches disabled
            (equivalent to ``docker build --no-cache``).
        """
        tag = tag or self.image_tag

        if not force:
            try:
                self.client.images.get(tag)
                print(f"[Task1Runner] Image '{tag}' already exists — skipping build.", flush=True)
                return
            except docker.errors.ImageNotFound:
                pass

        nocache = " (no-cache)" if force else ""
        print(f"[Task1Runner] Building image '{tag}' from {self.dockerfile_dir}{nocache} …", flush=True)
        _image, logs = self.client.images.build(
            path=str(self.dockerfile_dir),
            tag=tag,
            nocache=force,
            rm=True,
        )
        for chunk in logs:
            if "stream" in chunk:
                sys.stdout.write(chunk["stream"])
                sys.stdout.flush()
        print(f"[Task1Runner] Image '{tag}' built successfully.", flush=True)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        mode: str,
        size: str = "small",
        *,
        threads: int = 8,
        non_zeros: int = 600,
        k_top: int = 15,
        k_graph: int = 32,
        k_ext: int = 32,
        eps_ext: float = 0.001,
        max_dist: int | str = 200,
        evp_k: int | str = 50,
        prune_worst: int = 16,
        no_recall: bool = False,
        output: str | None = None,
        graph: str | None = None,
        timeout_s: float | None = None,
    ) -> Task1Result:
        """
        Run deglib_evp_task1 in a Docker container and return structured results.

        The container is started with SISAP-compliant resource limits:
            --cpus=8, --memory=24g, --memory-swap=24g, --memory-swappiness=0

        Volume mounts applied automatically:
            <hf_cache_dir>    → /data:ro
            ./results         → /results:rw

        Parameters
        ----------
        mode:
            Benchmark mode passed to the binary (e.g. ``"mode4"``, ``"evp"``).
        size:
            Dataset size: ``"small"`` or ``"large"``.
        threads:
            ``--threads`` argument (default 8 = full SISAP allocation).
        non_zeros:
            ``--non-zeros`` (EVP sparsity parameter).
        k_top:
            ``--k-top`` (number of nearest neighbours to retrieve).
        k_graph:
            ``--k-graph`` (graph degree).
        k_ext:
            ``--k-ext`` (builder search size).
        eps_ext:
            ``--eps-ext`` (builder entry-search expansion coefficient).
        max_dist:
            ``--max-dist``. May be an int or a comma-separated string for sweeps.
        evp_k:
            ``--evpK``. May be an int or a comma-separated string for sweeps.
        prune_worst:
            ``--prune-worst``.
        no_recall:
            Pass ``--no-recall`` to skip ground-truth loading.
        output:
            ``--output <path>`` (path *inside the container*, e.g. /results/out.ivecs).
        graph:
            ``--graph <path>`` (path *inside the container*).
        """
        # ---- Ensure results directory exists --------------------------------
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # ---- Resolve dataset -------------------------------------------------
        # Resolve dataset directories, handling symlinks dynamically (e.g. Hugging Face on Unix).
        # We find a common ancestor of the local file and its target so relative symlinks remain valid inside the container.
        data_dir = self.get_data_dir(size).absolute()
        hdf5_filename = DATASET_FILES[size]
        local_path = (data_dir / hdf5_filename).absolute()
        real_path = local_path.resolve()

        common_ancestor = None
        if local_path != real_path:
            p1_parents = [local_path] + list(local_path.parents)
            for parent in p1_parents:
                try:
                    real_path.relative_to(parent)
                    if len(parent.parts) > 2:  # Avoid root mount (e.g., '/' or 'C:\')
                        common_ancestor = parent
                        break
                except ValueError:
                    continue

        if common_ancestor is not None:
            # Mount the common ancestor and reference the dataset relative to it
            mount_dir = common_ancestor
            rel_path = local_path.relative_to(common_ancestor)
            container_hdf5 = f"/data/{rel_path.as_posix()}"
        else:
            # Fallback: mount the real file's parent directory and use the real filename
            mount_dir = real_path.parent
            container_hdf5 = f"/data/{real_path.name}"

        # ---- Build CLI command -----------------------------------------------
        cmd: list[str] = [
            "task1",
            container_hdf5,
            mode,
            "--threads",    str(threads),
            "--non-zeros",  str(non_zeros),
            "--k-top",      str(k_top),
            "--k-graph",    str(k_graph),
            "--k-ext",      str(k_ext),
            "--eps-ext",    str(eps_ext),
            "--max-dist",   str(max_dist),
            "--evpK",       str(evp_k),
            "--prune-worst", str(prune_worst),
        ]
        if no_recall:
            cmd.append("--no-recall")
        if output:
            cmd += ["--output", output]
        if graph:
            cmd += ["--graph", graph]

        # ---- Volume mounts ---------------------------------------------------
        volumes: dict[str, dict[str, str]] = {
            str(mount_dir): {"bind": "/data", "mode": "ro"},
            str(self.results_dir): {"bind": "/results", "mode": "rw"},
        }

        # ---- Resource limits (SISAP-compliant) ------------------------------
        run_kwargs: dict[str, Any] = {
            "image":         self.image_tag,
            "command":       cmd,
            "volumes":       volumes,
            "detach":        True,
            "stderr":        True,
            "stdout":        True,
            # CPU limit
            "nano_cpus":     _NANO_CPUS,
            # Memory limits (no swap beyond RAM)
            "mem_limit":     _MEM_LIMIT,
            "memswap_limit": _MEM_SWAP,
        }

        print(
            f"[Task1Runner] Starting container - mode={mode!r}, size={size!r}, "
            f"threads={threads}, max_dist={max_dist}",
            flush=True,
        )
        print(
            f"[Task1Runner] Limits: cpus=8, memory=24g, swap=24g, swappiness=0",
            flush=True,
        )
        print(f"[Task1Runner] Data mount : {mount_dir} -> /data:ro", flush=True)
        print(f"[Task1Runner] Results    : {self.results_dir} -> /results:rw", flush=True)

        # ---- Start container and stream logs --------------------------------
        parser = Task1LogParser(echo=self.echo_logs)

        try:
            container = self.client.containers.run(**run_kwargs)
        except docker.errors.ImageNotFound:
            raise RuntimeError(
                f"Docker image '{self.image_tag}' not found. "
                "Run Task1Runner().build_image() first."
            )

        # Optional watchdog: kill the container if it runs past timeout_s. The
        # blocking logs() stream then ends and wait() returns a non-zero code,
        # so the caller sees a failed run instead of hanging forever (guards
        # against pathological build configs in a hyperparameter sweep).
        timed_out = threading.Event()
        watchdog: threading.Timer | None = None
        if timeout_s:
            def _kill_on_timeout() -> None:
                timed_out.set()
                try:
                    container.kill()
                except docker.errors.APIError:
                    pass
            watchdog = threading.Timer(timeout_s, _kill_on_timeout)
            watchdog.daemon = True
            watchdog.start()

        try:
            # Stream logs line by line as they arrive
            for raw_chunk in container.logs(stream=True, follow=True):
                line = raw_chunk.decode("utf-8", errors="replace")
                parser.feed(line)

            # Wait for container to finish and get exit code
            result_info = container.wait()
            exit_code: int = result_info.get("StatusCode", -1)

            if timed_out.is_set():
                print(
                    f"[Task1Runner] Container exceeded timeout of {timeout_s:.0f}s "
                    f"and was killed.",
                    file=sys.stderr,
                    flush=True,
                )

        finally:
            if watchdog is not None:
                watchdog.cancel()
            # Always remove the container
            try:
                container.remove(force=True)
            except docker.errors.APIError:
                pass

        result = parser.build_result(
            mode=mode,
            dataset_size=size,
            exit_code=exit_code,
        )

        if exit_code != 0:
            print(
                f"[Task1Runner] Container exited with code {exit_code}.",
                file=sys.stderr,
                flush=True,
            )
            if parser.errors:
                for err in parser.errors:
                    print(f"  ERROR: {err}", file=sys.stderr)

        return result
