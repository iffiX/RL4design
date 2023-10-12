import os
from functools import partial
from typing import Callable
from launch.snapshot import init_or_load_launch_config
from navigator.trial import TrialRecord
from navigator.prompter import (
    Int,
    PromptApp,
    PromptChoiceDialog,
    PromptExecutableWithInput,
    PromptExecutableWithMultipleChoice,
)
from navigator.functions.single.draw_generation_process import (
    draw_generation_process,
)
from navigator.functions.single.draw_robots import draw_robots
from navigator.functions.single.compute_robot_metrics import (
    compute_robot_metrics,
    compute_all_robots_average_metrics,
)
from navigator.functions.single.visualize_robot_history import (
    visualize_selected_robot,
)
from navigator.functions.multi.draw_robot_metrics_curves import (
    draw_robot_metric_curves,
)
from navigator.functions.multi.draw_reward_curves import (
    draw_reward_curve,
    draw_separate_reward_curves,
)
from navigator.functions.multi.draw_volume_task_result import (
    draw_volume_task_result,
)
from navigator.functions.multi.draw_voxcraft_task_result import (
    draw_voxcraft_task_result,
)


def find_directories(
    root_dir_of_trials: str, trial_filter: Callable[[str], bool] = None
):
    trial_dirs = []
    if not os.path.exists(root_dir_of_trials):
        return trial_dirs
    source = os.listdir(root_dir_of_trials)
    source.sort(key=lambda x: os.path.getmtime(os.path.join(root_dir_of_trials, x)))
    for root_dir_of_trial in source:
        try:
            if os.path.isdir(os.path.join(root_dir_of_trials, root_dir_of_trial)) and (
                not trial_filter or trial_filter(root_dir_of_trial)
            ):
                sub_dir = [
                    sdir
                    for sdir in os.listdir(
                        os.path.join(root_dir_of_trials, root_dir_of_trial)
                    )
                    if os.path.isdir(
                        os.path.join(root_dir_of_trials, root_dir_of_trial, sdir)
                    )
                ][0]
                trial_dirs.append(
                    os.path.join(root_dir_of_trials, root_dir_of_trial, sub_dir)
                )
        except:
            continue
    return trial_dirs


def navigator_main(save_location: str = None):
    config = init_or_load_launch_config()
    all_trial_dirs = find_directories(save_location or config["save_location"])
    if len(all_trial_dirs) == 0:
        print(
            f"No trial result found under specified save location: {config['save_location']}"
        )
        return

    all_trial_records = []
    for trial_dir in all_trial_dirs:
        try:
            record = TrialRecord(trial_dir)
            all_trial_records.append(record)
        except:
            continue

    app = PromptApp(
        PromptChoiceDialog(
            description="",
            choices=[
                PromptChoiceDialog(
                    description="Draw metrics for multiple trials",
                    choices=[
                        PromptExecutableWithMultipleChoice(
                            description="Draw aggregated reward curve for multiple trials",
                            execute=draw_reward_curve,
                            choices=[
                                (
                                    f"{trial_record.trial_dir}\n"
                                    f"    comment: {' '.join(trial_record.comment)}\n"
                                    f"    reward: {trial_record.max_reward:.3f}",
                                    trial_record,
                                )
                                for trial_record in all_trial_records
                            ],
                        ),
                        PromptExecutableWithMultipleChoice(
                            description="Draw separate reward curves for each trial",
                            execute=draw_separate_reward_curves,
                            choices=[
                                (
                                    f"{trial_record.trial_dir}\n"
                                    f"    comment: {' '.join(trial_record.comment)}\n"
                                    f"    reward: {trial_record.max_reward:.3f}",
                                    trial_record,
                                )
                                for trial_record in all_trial_records
                            ],
                        ),
                        PromptExecutableWithMultipleChoice(
                            description="Draw aggregated robot metric curves for multiple trials",
                            execute=draw_robot_metric_curves,
                            choices=[
                                (
                                    f"{trial_record.trial_dir}\n"
                                    f"    comment: {' '.join(trial_record.comment)}\n"
                                    f"    reward: {trial_record.max_reward:.3f}",
                                    trial_record,
                                )
                                for trial_record in all_trial_records
                            ],
                        ),
                        PromptExecutableWithMultipleChoice(
                            description="Draw volume task result (For reproducing figures in paper)",
                            execute=draw_volume_task_result,
                            choices=[
                                (
                                    f"{trial_record.trial_dir}\n"
                                    f"    comment: {' '.join(trial_record.comment)}\n"
                                    f"    reward: {trial_record.max_reward:.3f}",
                                    trial_record,
                                )
                                for trial_record in all_trial_records
                            ],
                        ),
                        PromptExecutableWithMultipleChoice(
                            description="Draw voxcraft task result (For reproducing figures in paper)",
                            execute=draw_voxcraft_task_result,
                            choices=[
                                (
                                    f"{trial_record.trial_dir}\n"
                                    f"    comment: {' '.join(trial_record.comment)}\n"
                                    f"    reward: {trial_record.max_reward:.3f}",
                                    trial_record,
                                )
                                for trial_record in all_trial_records
                            ],
                        ),
                    ],
                ),
                PromptChoiceDialog(
                    description="Draw metrics for single trial",
                    choices=[
                        PromptChoiceDialog(
                            description=f"{trial_record.trial_dir}\n"
                            f"    comment: {' '.join(trial_record.comment)}\n"
                            f"    reward: {trial_record.max_reward:.3f}",
                            choices=[
                                PromptExecutableWithInput(
                                    "draw generation process figure for best robot from epoch...",
                                    partial(draw_generation_process, trial_record),
                                    prompt_input=f"show which epoch, from 1 to {trial_record.epochs[-1]}? "
                                    f"(-1 for best epoch)",
                                    input_formats=[Int()],
                                ),
                                PromptExecutableWithInput(
                                    "draw robots from epoch...",
                                    partial(draw_robots, trial_record),
                                    prompt_input=f"show which epoch, from 1 to {trial_record.epochs[-1]}? "
                                    f"(-1 for best epoch) and show how many robots?",
                                    input_formats=[Int(), Int(optional=True)],
                                ),
                                PromptExecutableWithInput(
                                    "compute metrics of selected robot from epoch...",
                                    partial(compute_robot_metrics, trial_record),
                                    prompt_input=f"show which epoch, from 1 to {trial_record.epochs[-1]}? "
                                    f"(-1 for best epoch), and which robot? (from best to worst, starting from 0)",
                                    input_formats=[Int(), Int()],
                                ),
                                PromptExecutableWithInput(
                                    "compute metrics of all robots from epoch...",
                                    partial(
                                        compute_all_robots_average_metrics,
                                        trial_record,
                                    ),
                                    prompt_input=f"show which epoch, from 1 to {trial_record.epochs[-1]}? "
                                    f"(-1 for best epoch)",
                                    input_formats=[Int()],
                                ),
                                PromptExecutableWithInput(
                                    "visualize history of selected robot from epoch...",
                                    partial(visualize_selected_robot, trial_record),
                                    prompt_input=f"show which epoch, from 1 to {trial_record.epochs[-1]}? "
                                    f"(-1 for best epoch)",
                                    input_formats=[Int()],
                                ),
                            ],
                        )
                        for trial_record in all_trial_records
                    ],
                    prompt_title="Trials",
                ),
            ],
        )
    )
    app.run()
