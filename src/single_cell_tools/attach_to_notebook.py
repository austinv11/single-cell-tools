import glob
import json
import os
import urllib.error
import urllib.request

import click
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text


def _find_kernel_connection_file(notebook_path: str) -> tuple[str, str] | None:
    """
    Query each running Jupyter server's sessions API to find the kernel
    UUID associated with the given notebook path, then return a tuple of
    (connection_file_path, execution_state).
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
                    kernel = session["kernel"]
                    kernel_id = kernel["id"]
                    execution_state = kernel.get("execution_state", "idle")
                    click.echo(f"Kernel ID: {kernel_id}")
                    return find_connection_file(kernel_id), execution_state

        except Exception:
            continue

    return None


def _notebook_cell_stats(notebook_path: str) -> tuple[int, int, str]:
    """
    Return (total_code_cells, max_execution_count, kernel_language) from the notebook file.
    max_execution_count reflects how many cell executions have occurred so far.
    kernel_language is the Pygments lexer name (e.g. "python" or "r").
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
        lang = nb.get("metadata", {}).get("kernelspec", {}).get("language", "python").lower()
        return total, max_ec, lang
    except Exception:
        return 0, 0, "python"


def _print_rich_output(console: Console, data: dict) -> None:
    """
    Render a Jupyter output data bundle, preferring HTML (via markdownify) over plain text.
    """
    html = data.get("text/html", "")
    if html:
        try:
            from markdownify import markdownify
            from rich.markdown import Markdown
            md = markdownify(html, heading_style="ATX").strip()
            if md:
                console.print(Markdown(md))
                return
        except ImportError:
            pass

    text = data.get("text/plain", "")
    if isinstance(text, list):
        text = "".join(text)
    if text:
        console.print(text, markup=False)


def _print_code_block(console: Console, code: str | list, execution_count: int | str, language: str) -> None:
    """Render a syntax-highlighted code block with execution count as title."""
    if isinstance(code, list):
        code = "".join(code)
    if not code.strip():
        return
    console.print(Panel(
        Syntax(code, language, theme="monokai", word_wrap=True),
        title=f"[dim]In [{execution_count}][/dim]",
        title_align="left",
        border_style="dim",
        padding=(0, 1),
    ))


def _find_likely_executing_cell(notebook_path: str) -> tuple[str, list, int | str] | None:
    """
    Scan the notebook to find the cell most likely currently executing.

    Strategy A: code cell with execution_count set but empty outputs (started, no output yet).
    Strategy B: first code cell after the last cell with non-empty outputs.
    Returns (source, outputs, execution_count) or None.
    """
    try:
        with open(notebook_path) as f:
            nb = json.load(f)
    except Exception:
        return None

    cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]

    # Strategy A: has execution_count (kernel assigned it) but outputs are empty
    in_progress = [c for c in cells if c.get("execution_count") and not c.get("outputs")]
    if in_progress:
        candidate = max(in_progress, key=lambda c: c["execution_count"])
        source = candidate.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        return source, candidate.get("outputs", []), candidate["execution_count"]

    # Strategy B: first cell after the last cell with non-empty outputs
    last_complete_idx = -1
    for i, c in enumerate(cells):
        if c.get("outputs"):
            last_complete_idx = i
    if last_complete_idx + 1 < len(cells):
        candidate = cells[last_complete_idx + 1]
        source = candidate.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        ec = candidate.get("execution_count") or "?"
        return source, candidate.get("outputs", []), ec

    return None


def _print_last_cell_output(
    console: Console, notebook_path: str, language: str, is_busy: bool = False
) -> None:
    """
    Read the notebook file and print the outputs of the last code cell
    that has non-empty outputs, so the user sees recent output immediately
    upon attaching.  When is_busy is True, also show the likely currently
    executing cell.
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

    if last_cell_with_output is not None:
        exec_count = last_cell_with_output.get("execution_count") or "?"
        console.print(f"[dim]--- Last cell output \\[execution count: {exec_count}] ---[/dim]")
        _print_code_block(console, last_cell_with_output.get("source", ""), exec_count, language)
        for output in last_cell_with_output["outputs"]:
            output_type = output.get("output_type", "")
            if output_type == "stream":
                text = output.get("text", "")
                if isinstance(text, list):
                    text = "".join(text)
                console.print(text, end="", markup=False)
            elif output_type in ("execute_result", "display_data"):
                _print_rich_output(console, output.get("data", {}))
            elif output_type == "error":
                traceback_lines = output.get("traceback", [])
                console.print("\n".join(traceback_lines), style="red", markup=False)
        console.print("[dim]--- End of last cell output ---[/dim]")

    if is_busy:
        executing = _find_likely_executing_cell(notebook_path)
        if executing is not None:
            source, outputs, ec = executing
            console.print("[dim]--- Currently executing cell ---[/dim]")
            _print_code_block(console, source, "...", language)
            for output in outputs:
                output_type = output.get("output_type", "")
                if output_type == "stream":
                    text = output.get("text", "")
                    if isinstance(text, list):
                        text = "".join(text)
                    console.print(text, end="", markup=False)
                elif output_type in ("execute_result", "display_data"):
                    _print_rich_output(console, output.get("data", {}))
                elif output_type == "error":
                    traceback_lines = output.get("traceback", [])
                    console.print("\n".join(traceback_lines), style="red", markup=False)
            console.print("[dim]--- (live outputs will stream below) ---[/dim]")


# Status colors per kernel execution_state value.
_STATE_STYLE = {
    "idle":     ("●", "bold green"),
    "busy":     ("●", "bold yellow"),
    "starting": ("●", "bold blue"),
}


def _process_stream(buf: str, text: str) -> tuple[str, list[str]]:
    """
    Handle carriage-return overwrite semantics (used by tqdm and similar).

    Returns (new_buf, completed_lines) where:
    - new_buf is the current in-progress line (not yet terminated by \\n)
    - completed_lines have a trailing \\n and are ready to print
    """
    completed: list[str] = []
    combined = buf + text
    current = ""
    i = 0
    while i < len(combined):
        ch = combined[i]
        if ch == "\r":
            if i + 1 < len(combined) and combined[i + 1] == "\n":
                completed.append(current + "\n")
                current = ""
                i += 2
            else:
                current = ""  # carriage return: overwrite the current line
                i += 1
        elif ch == "\n":
            completed.append(current + "\n")
            current = ""
            i += 1
        else:
            current += ch
            i += 1
    return current, completed


def _make_status_bar(kernel_state: str, cells_executed: int, total_cells: int) -> Text:
    symbol, style = _STATE_STYLE.get(kernel_state, ("●", "bold red"))

    bar = Text("\n")
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

    result = _find_kernel_connection_file(notebook_path)
    if result is None:
        console.print(
            "[red]Could not find a running kernel for this notebook.\n"
            "Make sure the notebook is open and running in Jupyter.[/red]"
        )
        return

    connection_file, kernel_state = result
    total_cells, cells_executed, language = _notebook_cell_stats(notebook_path)
    if kernel_state == "busy" and cells_executed > 0:
        cells_executed += 1
    _print_last_cell_output(console, notebook_path, language, is_busy=(kernel_state == "busy"))
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

    stream_buf = ""  # in-progress line (not yet \n-terminated; e.g. a tqdm bar)

    def _live_renderable() -> Group:
        status = _make_status_bar(kernel_state, cells_executed, total_cells)
        if stream_buf:
            return Group(Text(stream_buf), status)
        return Group(status)

    try:
        with Live(
            _live_renderable(),
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
                    if kernel_state == "dead":
                        console.print("[red]Kernel has been closed.[/red]")
                        return

                elif msg_type == "execute_input":
                    # Kernel just started executing a cell; bump the counter.
                    ec = content.get("execution_count") or 0
                    cells_executed = max(cells_executed, ec)
                    _print_code_block(console, content.get("code", ""), ec, language)

                elif msg_type == "stream":
                    stream_buf, completed = _process_stream(stream_buf, content["text"])
                    for line in completed:
                        console.print(line, end="", markup=False)

                elif msg_type == "execute_result":
                    _print_rich_output(console, content.get("data", {}))

                elif msg_type == "display_data":
                    _print_rich_output(console, content.get("data", {}))

                elif msg_type == "error":
                    console.print("\n".join(content.get("traceback", [])), style="red", markup=False)

                elif msg_type == "clear_output":
                    console.clear()

                live.update(_live_renderable())

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