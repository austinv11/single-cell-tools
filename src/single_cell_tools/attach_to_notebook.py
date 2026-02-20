import glob
import json
import os
import urllib.error
import urllib.request

import click
from rich.console import Console
from rich.live import Live
from rich.text import Text


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


def _notebook_cell_stats(notebook_path: str) -> tuple[int, int]:
    """
    Return (total_code_cells, max_execution_count) from the notebook file.
    max_execution_count reflects how many cell executions have occurred so far.
    """
    try:
        with open(notebook_path) as f:
            nb = json.load(f)
        code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        total = len(code_cells)
        max_ec = max(
            (c.get("execution_count") or 0 for c in code_cells),
            default=0,
        )
        return total, max_ec
    except Exception:
        return 0, 0


def _print_last_cell_output(console: Console, notebook_path: str) -> None:
    """
    Read the notebook file and print the outputs of the last code cell
    that has non-empty outputs, so the user sees recent output immediately
    upon attaching.
    """
    try:
        with open(notebook_path) as f:
            nb = json.load(f)
    except Exception:
        return

    last_cell_with_output = None
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code" and cell.get("outputs"):
            last_cell_with_output = cell

    if last_cell_with_output is None:
        return

    exec_count = last_cell_with_output.get("execution_count") or "?"
    console.print(f"[dim]--- Last cell output \\[execution count: {exec_count}] ---[/dim]")
    for output in last_cell_with_output["outputs"]:
        output_type = output.get("output_type", "")
        if output_type == "stream":
            text = output.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            console.print(text, end="")
        elif output_type in ("execute_result", "display_data"):
            data = output.get("data", {})
            text = data.get("text/plain", "")
            if isinstance(text, list):
                text = "".join(text)
            if text:
                console.print(text)
        elif output_type == "error":
            traceback_lines = output.get("traceback", [])
            console.print("\n".join(traceback_lines), style="red")
    console.print("[dim]--- End of last cell output ---[/dim]")


# Status colors per kernel execution_state value.
_STATE_STYLE = {
    "idle":     ("●", "bold green"),
    "busy":     ("●", "bold yellow"),
    "starting": ("●", "bold blue"),
}


def _make_status_bar(kernel_state: str, cells_executed: int, total_cells: int) -> Text:
    symbol, style = _STATE_STYLE.get(kernel_state, ("●", "bold red"))

    bar = Text()
    bar.append(" Kernel: ", style="bold")
    bar.append(f"{symbol} {kernel_state}", style=style)
    bar.append("  │  ", style="dim")
    bar.append("Cells: ", style="bold")

    # Progress bar
    width = 20
    filled = int(width * cells_executed / total_cells) if total_cells else 0
    filled = min(filled, width)
    bar.append("█" * filled, style="bold cyan")
    bar.append("░" * (width - filled), style="dim")
    bar.append(f" {cells_executed}/{total_cells}", style="bold")

    return bar


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

    console = Console(highlight=False)
    console.print(f"Finding kernel for notebook: [bold]{notebook_path}[/bold]")

    connection_file = _find_kernel_connection_file(notebook_path)
    if connection_file is None:
        console.print(
            "[red]Could not find a running kernel for this notebook.\n"
            "Make sure the notebook is open and running in Jupyter.[/red]"
        )
        return

    total_cells, cells_executed = _notebook_cell_stats(notebook_path)
    _print_last_cell_output(console, notebook_path)
    console.print(
        "Attaching to kernel "
        "([bold]Ctrl+C[/bold] interrupt & detach, [bold]Ctrl+Z[/bold] detach only)..."
    )

    import signal

    class _DetachOnly(Exception):
        pass

    old_sigtstp = signal.signal(signal.SIGTSTP, lambda s, f: (_ for _ in ()).throw(_DetachOnly()))

    client = jupyter_client.BlockingKernelClient(connection_file=connection_file)
    client.load_connection_file()
    client.start_channels()

    kernel_state = "idle"

    try:
        with Live(
            _make_status_bar(kernel_state, cells_executed, total_cells),
            console=console,
            refresh_per_second=8,
            transient=False,
        ) as live:
            while True:
                msg = client.get_iopub_msg()
                msg_type = msg["msg_type"]
                content = msg.get("content", {})

                if msg_type == "status":
                    kernel_state = content.get("execution_state", kernel_state)

                elif msg_type == "execute_input":
                    # Kernel just started executing a cell; bump the counter.
                    ec = content.get("execution_count") or 0
                    cells_executed = max(cells_executed, ec)

                elif msg_type == "stream":
                    console.print(content["text"], end="")

                elif msg_type == "execute_result":
                    console.print(content["data"].get("text/plain", ""))

                elif msg_type == "display_data":
                    text = content.get("data", {}).get("text/plain", "")
                    if text:
                        console.print(text)

                elif msg_type == "error":
                    console.print("\n".join(content.get("traceback", [])), style="red")

                elif msg_type == "clear_output":
                    console.clear()

                elif msg_type == "comm_close" or (
                    msg_type == "status" and content.get("execution_state") == "dead"
                ):
                    console.print("[red]Kernel has been closed.[/red]")
                    return

                live.update(_make_status_bar(kernel_state, cells_executed, total_cells))

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupting kernel and detaching...[/yellow]")
        try:
            msg = client.session.msg("interrupt_request", {})
            client.control_channel.send(msg)
        except Exception as e:
            console.print(f"[red]Warning: could not send interrupt to kernel: {e}[/red]")

    except _DetachOnly:
        console.print("\n[yellow]Detaching from kernel...[/yellow]")

    finally:
        signal.signal(signal.SIGTSTP, old_sigtstp)
        client.stop_channels()


if __name__ == "__main__":
    main()