import sys

import click


@click.command()
@click.argument('notebook_path', type=click.Path(exists=True))
def main(notebook_path: str):
    """
    Find and identify the running jupyter kernel associated with a notebook,
    then attach to the current process and follow the kernel's output.

    This allows the user to see the output of the notebook in real time.

    :param notebook_path: Path to the Jupyter notebook to attach to.
    """
    try:
        import jupyter_client
        import nbformat
    except:
        click.echo("Make sure this is run in the same environment that Jupyter is installed to.")

    # Find the kernel associated with the notebook
    click.echo(f"Finding kernel for notebook: {notebook_path}")
    nb = nbformat.read(notebook_path)
    kernel_id = nb['metadata']['kernelspec']['name']
    click.echo(f"Kernel ID: {kernel_id}")

    def check_kernel_is_busy(kernel_manager):
        for kernel in kernel_manager.list_kernels():
            if kernel['id'] == kernel_id:
                return kernel['execution_state'] == 'busy'
        return False

    # Verify the kernel is running
    kernel_manager = jupyter_client.KernelManager()
    kernel_manager.load_connection_file(kernel_id)
    if not check_kernel_is_busy(kernel_manager):
        click.echo("Kernel is not currently running. The notebook must have a currently running cell.")
        return

    click.echo("Attaching to kernel, interrupt the process to detach...")

    connection_file = jupyter_client.find_connection_file(kernel_id)
    client = jupyter_client.BlockingKernelClient(connection_file=connection_file)
    client.load_connection_file()
    client.start_channels()

    try:
        while True:
            msg = client.get_iopub_msg()
            if msg['msg_type'] == 'stream':
                print(msg['content']['text'], end='', flush=True)
            elif msg['msg_type'] == 'execute_result':
                print("==== Execute Result ====", flush=True)
                print(msg['content']['data']['text/plain'], flush=True)
            elif msg['msg_type'] == 'error':
                print('\n'.join(msg['content']['traceback']), flush=True, file=sys.stderr)
            elif msg['msg_type'] == 'status' and msg['content']['execution_state'] == 'idle':
                click.echo("Kernel is idle. Waiting for next cell execution...")
            elif msg['msg_type'] == 'close':
                click.echo("Kernel has been closed.")
                break

    except KeyboardInterrupt:
        click.echo("Detaching from kernel...")
    finally:
        client.stop_channels()


if __name__ == '__main__':
    main()