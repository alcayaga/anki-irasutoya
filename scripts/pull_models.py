#!/usr/bin/env python3
"""Declarative local Ollama model installer for anki-irasutoya.

Reads model definitions from models.yml, compares against locally installed Ollama
models via the /api/tags endpoint, and pulls any missing models.
"""
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Sequence, Set

logger = logging.getLogger("pull_models")

DEFAULT_MODELS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models.yml",
)


def load_models_manifest(path: str = DEFAULT_MODELS_PATH) -> Dict[str, Any]:
    """Load model definitions from YAML or fallback JSON manifest.

    Args:
        path: Path to models.yml or models.json.

    Returns:
        Dictionary containing manifest configuration.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
        ValueError: If manifest contents are invalid.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Models manifest not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        import yaml
        try:
            data = yaml.safe_load(content)
        except Exception as exc:
            raise ValueError(f"Failed to parse YAML manifest {path}: {exc}") from exc
    except ImportError:
        # Fallback to JSON parsing if YAML library is not available
        try:
            data = json.loads(content)
        except Exception as exc:
            raise ValueError(f"Failed to parse manifest {path} as JSON/YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Root manifest structure in {path} must be a dictionary")
    return data


def get_installed_ollama_models(
    ollama_url: str = "http://localhost:11434",
    timeout: float = 5.0,
) -> Set[str]:
    """Retrieve the set of installed model identifiers from Ollama /api/tags.

    Args:
        ollama_url: Base URL of the Ollama server.
        timeout: Network probe timeout in seconds.

    Returns:
        Set of exact model names installed locally.
    """
    parsed = urllib.parse.urlparse(ollama_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else "http://localhost:11434"
    tags_url = f"{base_url.rstrip('/')}/api/tags"

    req = urllib.request.Request(tags_url, headers={"User-Agent": "anki-irasutoya/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.error("Failed to query Ollama at %s: %s", tags_url, exc)
        return set()

    installed: Set[str] = set()
    models = data.get("models", [])
    if isinstance(models, list):
        for m in models:
            if isinstance(m, dict) and "name" in m:
                name = str(m["name"]).strip()
                installed.add(name)
                # Also index canonical tag (e.g. 'model:latest' -> 'model')
                if name.endswith(":latest"):
                    installed.add(name[:-len(":latest")])
    return installed


def pull_single_model(
    model_name: str,
    ollama_url: str = "http://localhost:11434",
    stream_progress: bool = True,
) -> bool:
    """Pull a model into Ollama via CLI subprocess or HTTP /api/pull.

    Args:
        model_name: Name and tag of the model to pull (e.g. 'gemma4:26b').
        ollama_url: Base URL of the Ollama server.
        stream_progress: Whether to stream progress indicators.

    Returns:
        True if the model was pulled successfully, False otherwise.
    """
    ollama_bin = shutil.which("ollama")
    if ollama_bin:
        logger.info("Pulling model '%s' via 'ollama pull'...", model_name)
        try:
            env = {**os.environ, "OLLAMA_HOST": ollama_url}
            res = subprocess.run([ollama_bin, "pull", model_name], check=True, env=env)
            return res.returncode == 0
        except subprocess.CalledProcessError as exc:
            logger.error("'ollama pull %s' failed: %s", model_name, exc)
            return False

    # HTTP API streaming fallback if ollama CLI is not found on PATH
    logger.info("Ollama CLI not found on PATH. Pulling '%s' via HTTP API...", model_name)
    parsed = urllib.parse.urlparse(ollama_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else "http://localhost:11434"
    pull_url = f"{base_url.rstrip('/')}/api/pull"

    payload = json.dumps({"name": model_name, "stream": stream_progress}).encode("utf-8")
    req = urllib.request.Request(
        pull_url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "anki-irasutoya/1.0"},
    )
    succeeded = False
    try:
        with urllib.request.urlopen(req, timeout=7200.0) as resp:
            for line in resp:
                clean_line = line.decode("utf-8", errors="replace").strip()
                if not clean_line:
                    continue
                try:
                    event = json.loads(clean_line)
                except Exception:
                    continue
                if not isinstance(event, dict):
                    continue
                if event.get("error"):
                    logger.error("Ollama pull error for '%s': %s", model_name, event["error"])
                    return False
                status = event.get("status", "")
                if status == "success":
                    succeeded = True
                completed = event.get("completed", 0) or 0
                total = event.get("total", 0) or 0
                if total > 0 and stream_progress:
                    pct = (completed / total) * 100
                    print(f"\r  {status}: {pct:.1f}% ({completed}/{total})", end="", flush=True)
                elif status and stream_progress:
                    print(f"\r  {status}...", end="", flush=True)
            if stream_progress:
                print()
        if not succeeded:
            logger.error("Ollama pull for '%s' ended without a success status", model_name)
        return succeeded
    except Exception as exc:
        logger.error("HTTP pull request failed for '%s': %s", model_name, exc)
        return False


def pull_declared_models(
    manifest_path: str = DEFAULT_MODELS_PATH,
    ollama_url: Optional[str] = None,
    dry_run: bool = False,
    force: bool = False,
) -> bool:
    """Pull all missing models defined in models.yml.

    Args:
        manifest_path: Path to models.yml.
        ollama_url: Base URL of Ollama service. If omitted, uses endpoint in manifest or http://localhost:11434.
        dry_run: If True, inspects status without pulling.
        force: If True, pulls even if already installed.

    Returns:
        True if all models are present or pulled successfully, False otherwise.
    """
    manifest = load_models_manifest(manifest_path)
    ollama_cfg = manifest.get("ollama", {}) if isinstance(manifest.get("ollama"), dict) else manifest
    raw_ep = (
        ollama_url
        or ollama_cfg.get("endpoint")
        or manifest.get("endpoint")
        or "http://localhost:11434"
    )
    parsed_ep = urllib.parse.urlparse(raw_ep)
    ep_path = parsed_ep.path.rstrip("/")
    if ep_path.endswith("/api/pull") or ep_path.endswith("/api/chat") or ep_path.endswith("/api/tags"):
        ep_path = ep_path.rsplit("/", 2)[0]
    elif ep_path.endswith("/api"):
        ep_path = ep_path[:-len("/api")]
    endpoint = f"{parsed_ep.scheme}://{parsed_ep.netloc}{ep_path}" if parsed_ep.netloc else "http://localhost:11434"
    models_list = ollama_cfg.get("models", []) if "models" in ollama_cfg else manifest.get("models", [])

    if not isinstance(models_list, list) or not models_list:
        logger.warning("No models declared in manifest %s", manifest_path)
        return True

    installed = get_installed_ollama_models(endpoint)
    logger.info("Checking %d declared models against Ollama at %s...", len(models_list), endpoint)

    all_success = True
    for item in models_list:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        role = item.get("role", "general")
        desc = item.get("description", "")
        size_gb = item.get("size_gb", 0)

        if not name:
            continue

        is_present = name in installed or (f"{name}:latest" in installed)
        if is_present and not force:
            logger.info("✓ Model '%s' (%s, ~%.1f GB) is already installed.", name, role, size_gb)
            continue

        action_msg = "Re-pulling" if force else "Missing"
        logger.info(
            "➔ %s model '%s' (%s, ~%.1f GB): %s",
            action_msg,
            name,
            role,
            size_gb,
            desc,
        )

        if dry_run:
            logger.info("  [dry-run] Would pull '%s'", name)
            continue

        ok = pull_single_model(name, ollama_url=endpoint)
        if not ok:
            all_success = False
            logger.error("Failed to pull declared model '%s'", name)
        else:
            logger.info("✓ Successfully installed '%s'.", name)

    return all_success


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for pulling declarative Ollama models."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="Declaratively pull local Ollama models defined in models.yml for anki-irasutoya."
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MODELS_PATH,
        help=f"Path to models.yml (default: {DEFAULT_MODELS_PATH})",
    )
    parser.add_argument(
        "--ollama-url",
        default=None,
        help="Ollama base URL (default: manifest endpoint, else http://localhost:11434)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check model status without pulling",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-pulling models even if already present",
    )

    args = parser.parse_args(argv)
    ok = pull_declared_models(
        manifest_path=args.manifest,
        ollama_url=args.ollama_url,
        dry_run=args.dry_run,
        force=args.force,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
