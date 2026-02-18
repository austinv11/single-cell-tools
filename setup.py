import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from setuptools import setup
from setuptools.command.build_scripts import build_scripts as _build_scripts

# Entry-point wrapper scripts to compile.
SCRIPTS = [
    "scripts/compress_h5ad",
    "scripts/attach_to_notebook",
]


class NuitkaBuildScripts(_build_scripts):
    """build_scripts override: compiles wrapper scripts to standalone binaries via Nuitka.

    Falls back to plain Python scripts if Nuitka is not installed or compilation fails.
    Nuitka must be available in the build environment (listed under [build-system].requires).
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

        # Make the project's own source available to Nuitka when it follows imports.
        build_py = self.get_finalized_command("build_py")
        build_lib = getattr(build_py, "build_lib", None)

        build_dir = Path(self.build_dir)

        for script in SCRIPTS:
            script_name = Path(script).name
            script_path = build_dir / script_name

            if not script_path.exists():
                continue

            print(f"Compiling {script_name} with Nuitka (--onefile) ...")

            with tempfile.TemporaryDirectory() as tmpdir:
                # Nuitka appends .exe on Windows automatically when -o has no extension,
                # so we name the output file explicitly to match the script name.
                if sys.platform == "win32":
                    output_name = script_name + ".exe"
                else:
                    output_name = script_name

                output_path = Path(tmpdir) / output_name

                # Extend PYTHONPATH so Nuitka can import the freshly-built package.
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
    cmdclass={"build_scripts": NuitkaBuildScripts},
)