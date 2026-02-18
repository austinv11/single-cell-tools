import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from setuptools import setup
from setuptools.command.build import build as _build

# Entry-point wrapper scripts to compile.
SCRIPTS = [
    "scripts/compress_h5ad",
    "scripts/attach_to_notebook",
]


class NuitkaBuild(_build):
    """build override: compiles wrapper scripts to standalone binaries via Nuitka.

    Hooks into the standard `build` command (which always exists in setuptools)
    rather than `build_scripts`, which is not present in all setuptools versions.
    Falls back to plain Python scripts if Nuitka is not installed or compilation fails.
    """

    def run(self):
        super().run()
        self._try_nuitka_compile()

    def _try_nuitka_compile(self):
        try:
            import nuitka  # noqa: F401 – just verify it is importable
        except ImportError:
            print(
                "nuitka not found in build environment – "
                "scripts will be installed as plain Python scripts."
            )
            return

        # Locate the directory where build_scripts placed the processed scripts.
        build_scripts_cmd = self.get_finalized_command("build_scripts")
        build_dir = Path(build_scripts_cmd.build_dir)

        if not build_dir.exists():
            return

        # Extend PYTHONPATH so Nuitka can import the freshly-built package.
        build_py_cmd = self.get_finalized_command("build_py")
        build_lib = getattr(build_py_cmd, "build_lib", None)

        for script in SCRIPTS:
            script_name = Path(script).name
            script_path = build_dir / script_name

            if not script_path.exists():
                continue

            print(f"Compiling {script_name} with Nuitka (--onefile) ...")

            with tempfile.TemporaryDirectory() as tmpdir:
                # Name the output explicitly; Nuitka adds .exe on Windows automatically.
                output_name = script_name + (".exe" if sys.platform == "win32" else "")
                output_path = Path(tmpdir) / output_name

                env = os.environ.copy()
                if build_lib:
                    env["PYTHONPATH"] = (
                        str(build_lib) + os.pathsep + env.get("PYTHONPATH", "")
                    )

                cmd = [
                    sys.executable,
                    "-m",
                    "nuitka",
                    "--onefile",
                    "--assume-yes-for-downloads",
                    f"--output-dir={tmpdir}",
                    "-o",
                    str(output_path),
                    str(script_path),
                ]

                try:
                    subprocess.run(cmd, check=True, env=env)

                    if output_path.exists():
                        script_path.unlink()
                        shutil.copy2(str(output_path), str(script_path))
                        if sys.platform != "win32":
                            os.chmod(script_path, 0o755)
                        print(f"  OK: {script_name} compiled to native binary.")
                    else:
                        print(
                            f"  Warning: Nuitka finished but produced no output for "
                            f"{script_name}. Keeping Python script."
                        )

                except (subprocess.CalledProcessError, OSError) as exc:
                    print(
                        f"  Warning: Nuitka compilation failed for {script_name}: {exc}\n"
                        f"  Keeping plain Python script as fallback."
                    )


setup(
    scripts=SCRIPTS,
    cmdclass={"build": NuitkaBuild},
)