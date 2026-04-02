"""Built-in SSH tools for managing remote servers.

Provides two async tools always available to the agent:
  - ssh_exec(command, timeout)            — run a shell command on the remote server
  - ssh_upload_file(remote_path, content) — write a text file on the remote server

Credentials are read from environment variables:
  SSH_HOST  — server IP or hostname
  SSH_USER  — SSH username (default: root)
  SSH_PASS  — SSH password (optional, empty = key-based auth)
  SSH_KEY   — path to private key file on disk (optional)
  SSH_PORT  — SSH port (default: 22)

Variables are set via the save_env_var tool and stored in .env.
"""
from __future__ import annotations

import io
import os

import paramiko
import structlog

logger = structlog.get_logger()

_MAX_OUTPUT_CHARS = 8000


def _get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _make_client() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    return client


def _connect(client: paramiko.SSHClient) -> None:
    host = _get_env("SSH_HOST")
    if not host:
        raise ValueError("SSH_HOST не задан. Используй save_env_var('SSH_HOST', '...')")

    user = _get_env("SSH_USER", "root")
    password = _get_env("SSH_PASS") or None
    key_path = _get_env("SSH_KEY") or None
    port = int(_get_env("SSH_PORT", "22"))

    connect_kwargs: dict = dict(
        hostname=host,
        port=port,
        username=user,
        timeout=15,
        allow_agent=True,
        look_for_keys=True,
    )
    if password:
        connect_kwargs["password"] = password
    if key_path:
        connect_kwargs["key_filename"] = key_path

    logger.info("ssh_connect", host=host, port=port, user=user)
    client.connect(**connect_kwargs)


async def ssh_exec(command: str, timeout: int = 60) -> str:
    """Execute a shell command on the remote server via SSH and return the output.

    Reads SSH_HOST, SSH_USER, SSH_PASS (or SSH_KEY), SSH_PORT from environment.
    Set them first with save_env_var if not already configured.

    Args:
        command: Shell command to run, e.g. "docker ps" or "systemctl status nginx".
        timeout: Seconds to wait for the command to finish (default: 60).

    Returns:
        Combined stdout/stderr output with exit code, or an SSH error description.
    """
    client = _make_client()
    try:
        _connect(client)
        _, stdout, stderr = client.exec_command(command, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        rc = stdout.channel.recv_exit_status()

        parts = [f"[exit code: {rc}]"]
        if out:
            parts.append(f"STDOUT:\n{out.rstrip()}")
        if err.strip():
            parts.append(f"STDERR:\n{err.rstrip()}")

        result = "\n".join(parts)
        if len(result) > _MAX_OUTPUT_CHARS:
            result = (
                result[:_MAX_OUTPUT_CHARS]
                + f"\n... (обрезано, показано {_MAX_OUTPUT_CHARS} из {len(result)} символов)"
            )
        logger.info("ssh_exec_done", rc=rc, command=command[:80])
        return result

    except Exception as exc:
        logger.error("ssh_exec_error", error=str(exc), command=command[:80])
        return f"SSH ERROR: {exc}"
    finally:
        client.close()


async def ssh_upload_file(remote_path: str, content: str) -> str:
    """Write a text file to the remote server via SFTP.

    Creates parent directories automatically.
    Useful for deploying nginx configs, docker-compose.yml, systemd units, scripts, etc.

    Args:
        remote_path: Absolute path on the remote server, e.g. /etc/nginx/sites-available/myapp.
        content: Text content to write to the file.

    Returns:
        Success message or an error description.
    """
    client = _make_client()
    try:
        _connect(client)

        parent = remote_path.rsplit("/", 1)[0] if "/" in remote_path else ""
        if parent:
            _, stdout, _ = client.exec_command(f"mkdir -p {parent!r}")
            stdout.channel.recv_exit_status()

        sftp = client.open_sftp()
        file_bytes = io.BytesIO(content.encode("utf-8"))
        sftp.putfo(file_bytes, remote_path)
        sftp.close()

        logger.info("ssh_upload_done", path=remote_path, size=len(content))
        return f"Файл записан: {remote_path} ({len(content)} байт)"

    except Exception as exc:
        logger.error("ssh_upload_error", error=str(exc), path=remote_path)
        return f"SFTP ERROR: {exc}"
    finally:
        client.close()
