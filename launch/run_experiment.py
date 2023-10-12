import os
import sys
import shutil
import subprocess
from launch.snapshot import (
    init_or_load_launch_config,
    get_snapshot_comment_file,
    get_snapshot,
)


def run_experiment(
    experiment_py_relative_path: str,
):
    init_or_load_launch_config()
    comment_file, comment_dir = get_snapshot_comment_file()
    snapshot_dir = get_snapshot(code_only=False)

    with open(os.path.join(snapshot_dir, "LAUNCH_COMMAND.sh"), "w") as file:
        file.write(f"{sys.executable} {experiment_py_relative_path}")

    # Move comment file to the snapshot dir, so it will be saved by the
    # experiment runners when they call get_snapshot()
    shutil.copy2(comment_file, snapshot_dir)
    shutil.rmtree(comment_dir)

    # Launch python process from the snapshot dir
    command = [sys.executable, experiment_py_relative_path]

    process = None
    try:
        process = subprocess.Popen(
            command,
            cwd=snapshot_dir,
            env=os.environ.update({"PYTHONPATH": snapshot_dir}),
            start_new_session=True,
        )
        code = process.wait()
        print(f"Launch exited with code {code}")
        if code != 0:
            print(f"Inspect temp code directory {snapshot_dir}")
        else:
            print(f"Removing temp code directory {snapshot_dir}")
            shutil.rmtree(snapshot_dir)
    except KeyboardInterrupt:
        if process is not None:
            print("Keyboard interrupt received, killing instance")
            process.kill()
        print(f"Removing temp code directory {snapshot_dir}")
        shutil.rmtree(snapshot_dir)
