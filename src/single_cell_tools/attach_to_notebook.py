import glob
import json
import os
import sys
import urllib.error
import urllib.request

import click


def _find_kernel_connection_file(notebook_path: str) -> str | None:
    """
    Query each running Jupyter server's sessions API to find the kernel
    UUID associated with the given notebook path, then return the path to
    its connection file.
    """
    try:
        from jupyter_client import find_connection_file
        from jupyter_client.connect import jupyter_runtime_dir
    except ImportError:
        return None

    runtime_dir = jupyter_runtime_dir()
    abs_notebook = os.path.abspath(notebook_path)

    # Jupyter writes one *server-<pid>.json per running server instance.
    server_files = sorted(
        glob.glob(os.path.join(runtime_dir, "*server-*.json")),
        key=os.path.getmtime,
        reverse=True,
    )

    for server_file in server_files:
        try:
            with open(server_file) as f:
                server_info = json.load(f)

            url = server_info.get("url", "").rstrip("/")
            token = server_info.get("token", "")
            if not url:
                continue

            req = urllib.request.Request(
                f"{url}/api/sessions",
                headers={"Authorization": f"token {token}"},
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                sessions = json.loads(resp.read())

            root_dir = server_info.get("root_dir") or server_info.get("notebook_dir", "")

            for session in sessions:
                session_abs = os.path.abspath(
                    os.path.join(root_dir, session.get("path", ""))
                )
                if session_abs == abs_notebook:
                    kernel_id = session["kernel"]["id"]
                    click.echo(f"Kernel ID: {kernel_id}")
                    return find_connection_file(kernel_id)

        except Exception:
            continue

    return None


@click.command()
@click.argument("notebook_path", type=click.Path(exists=True))
def main(notebook_path: str):
    """
    Find and attach to the running Jupyter kernel associated with a notebook,
    then stream its output to the terminal in real time.
    """
    try:
        import jupyter_client
    except ImportError:
        click.echo("Make sure this is run in the same environment that Jupyter is installed to.")
        return

    click.echo(f"Finding kernel for notebook: {notebook_path}")

    connection_file = _find_kernel_connection_file(notebook_path)
    if connection_file is None:
        click.echo(
            "Could not find a running kernel for this notebook.\n"
            "Make sure the notebook is open and running in Jupyter."
        )
        return

    click.echo("Attaching to kernel (Ctrl+C to interrupt kernel, Ctrl+C twice to detach)...")

    client = jupyter_client.BlockingKernelClient(connection_file=connection_file)
    client.load_connection_file()
    client.start_channels()

    last_interrupt = 0.0

    try:
        while True:
            try:
                while True:
                    msg = client.get_iopub_msg()
                    if msg["msg_type"] == "stream":
                        print(msg["content"]["text"], end="", flush=True)
                    elif msg["msg_type"] == "execute_result":
                        print("==== Execute Result ====", flush=True)
                        print(msg["content"]["data"]["text/plain"], flush=True)
                    elif msg["msg_type"] == "error":
                        print("\n".join(msg["content"]["traceback"]), flush=True, file=sys.stderr)
                    elif msg["msg_type"] == "status" and msg["content"]["execution_state"] == "idle":
                        click.echo("Kernel is idle. Waiting for next cell execution...")
                    elif msg["msg_type"] == "close":
                        click.echo("Kernel has been closed.")
                        return

            except KeyboardInterrupt:
                import time
                now = time.monotonic()
                if now - last_interrupt < 2.0:
                    click.echo("\nDetaching from kernel...")
                    return
                last_interrupt = now
                click.echo("\nInterrupting kernel... (Ctrl+C again within 2 s to detach)")
                try:
                    msg = client.session.msg("interrupt_request", {})
                    client.control_channel.send(msg)
                except Exception as e:
                    click.echo(f"Warning: could not send interrupt to kernel: {e}")

    finally:
        client.stop_channels()


if __name__ == "__main__":
    main()