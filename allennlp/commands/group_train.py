import argparse
import logging
import os
import json

from allennlp.commands.subcommand import Subcommand
from allennlp.commands.train import train_model_from_file
from allennlp.common import Params

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class Group_Train(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Train the specified model on the specified dataset.'''
        subparser = parser.add_parser(name, description=description, help='Train by parameter files in a dir.')

        subparser.add_argument('param_path',
                               type=str,
                               help='path to parameter files describing the model to be trained')

        subparser.add_argument('-s', '--serialization-dir',
                               required=True,
                               type=str,
                               help='directory in which to save each model in a sub-dir and their logs')

        subparser.add_argument('--file-friendly-logging',
                               action='store_true',
                               default=False,
                               help='outputs tqdm status on separate lines and slows tqdm refresh rate')

        subparser.set_defaults(func=group_train_from_args)

        return subparser


def group_train_from_args(args: argparse.Namespace):
    """
    Just converts from an ``argparse.Namespace`` object to params.
    """
    train_model_from_files(args.param_path,
                          args.serialization_dir,
                          args.file_friendly_logging)


def train_model_from_files(param_path: str,
                          serialization_dir: str,
                          file_friendly_logging: bool = False) -> None:
    """
    load the params from a file and train a model, then the next.

    Parameters
    ----------
    param_path : ``str``
        A dir contains json parameter files specifying a group of AllenNLP experiment.
        'training_progress.json', which record training progress, will be created here.
    serialization_dir : ``str``
        The directory in which to save results and logs. A parameter file corresponds
        to the sub_serial_path with the same name.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we make our output more friendly to saved model files.  We just pass this
        along to :func:`train_model`.
    """
    if os.path.isabs(param_path) == False:
        logger.warn(f"param_path must be a absolute path")
        return

    if os.path.isfile(serialization_dir):
        logger.warn(f"serialization_dir must be a path, but you input a file")
        return

    param_files = os.listdir(param_path)  #listdir must use absolute path
    if len(param_files) == 0:
        logger.warn(f"At least one parameter file in the param_path directory")
        return

    for file in param_files:
        if os.path.isdir(file):
            param_files.remove(file) # exclude dir name
        if os.path.splitext(file)[1] not in ['.json', '.jsonnet']:
            param_files.remove(file)  # only include json and jsonnet
        if file == 'training_progress.json':
            param_files.remove(file)  # if training_progress.json exists, remove it.

    start_position  = check_progress_file(param_path)

    if start_position == 'All Finished':
        logger.info(f"param_path must be a absolute path")
        return

    if start_position is not None:
        idx = param_files.index(start_position)
        param_files = param_files[idx:]

    for file in param_files:
        if check_file_for_train(param_path, file):
            continue
        param_file_name = param_path + os.sep + file
        sub_serial_path = serialization_dir + os.sep + os.path.splitext(file)[0]  # XXX.json/.jsonnet
        if start_position is None:
            train_model_from_file(parameter_filename = param_file_name,
                                  serialization_dir = sub_serial_path,
                                  file_friendly_logging = file_friendly_logging)
        else:
            train_model_from_file(parameter_filename=param_file_name,
                                  serialization_dir=sub_serial_path,
                                  file_friendly_logging=file_friendly_logging,
                                  recover=True)
            start_position = None  # Only the first False needs recover, others train as usual

        update_progress_file(param_path) # update training_progress.json to record training state.


def check_progress_file(param_path) -> None:
    """
    check training_progress.json. It contains a Dict, the key
    is the name of the parameter file, and the value is whether
    (true/false) the training has been completed.
    If it exists, determine which file to recover training.
    If it doesn't exist, create it.

    Parameters
    ----------
    param_path : ``str``
        Where to check training_progress.json.
    """
    param_file_name = param_path + os.sep + 'training_progress.json'
    if os.path.exists(param_file_name) is not True:   # create training_progress.json at first time
        logger.info(f"Creating training_progress.json at {param_path}.")

        files = os.listdir(param_path)
        progress = {str(file):False for file in files}
        with open(param_file_name, 'w') as f:
            json.dump(progress, f)
        return None
    else:           # continue from which file
        logger.info(f"Recover group train.")
        with open(param_file_name, 'r') as f:
            progress = json.load(f)
            for file, bFinished in progress.items():
                if bFinished == False:
                    return file
        return 'All Finished'


def update_progress_file(param_path) -> None:
    """
    Change a training_progress.json item after a train.

    Parameters
    ----------
    param_path : ``str``
        Where to update training_progress.json.
    """
    progress = {}
    param_file_name = param_path + os.sep + 'training_progress.json'
    with open(param_file_name, 'r') as f:
        progress = json.load(f)
    for file, bFinished in progress.items():
        if bFinished == False:
            progress[file] = True
            break
    with open(param_file_name, 'w') as f:
        json.dump(progress, f)

def check_file_for_train(param_path, file) -> bool:
    """
    Check if the training task corresponding to this file has been completed.

    Parameters
    ----------
    param_path : ``str``
        Where to check training_progress.json.
    file : ``str``
        File name to check
    """
    param_file_name = param_path + os.sep + 'training_progress.json'
    with open(param_file_name, 'r') as f:
        progress = json.load(f)
    return progress[file]
