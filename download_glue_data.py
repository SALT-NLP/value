import os
import sys
import shutil
import argparse
import tempfile
import urllib.request
import zipfile

TASKS = ["CoLA", "SST", "QQP", "STS", "MNLI", "QNLI", "RTE", "WNLI"]
TASK2PATH = {"CoLA":'https://dl.fbaipublicfiles.com/glue/data/CoLA.zip',
             "SST":'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip',
             "MRPC":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-48f4-b431-7480817f1adc',
             "QQP":'https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip',
             "STS":'https://dl.fbaipublicfiles.com/glue/data/STS-B.zip',
             "MNLI":'https://dl.fbaipublicfiles.com/glue/data/MNLI.zip',
             "QNLI": 'https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip',
             "RTE":'https://dl.fbaipublicfiles.com/glue/data/RTE.zip',
             "WNLI":'https://dl.fbaipublicfiles.com/glue/data/WNLI.zip'}

def download_and_extract(task, data_dir):
    print("Downloading and extracting %s..." % task)
    data_file = "%s.zip" % task
    urllib.request.urlretrieve(TASK2PATH[task], data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)
    print("\tCompleted!")

def get_tasks(task_names):
    task_names = task_names.split(',')
    if "all" in task_names:
        tasks = TASKS
    else:
        tasks = []
        for task_name in task_names:
            assert task_name in TASKS, "Task %s not found!" % task_name
            tasks.append(task_name)
    return tasks

def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='directory to save data to', type=str, default='./data/GLUE')
    parser.add_argument('--tasks', help='tasks to download data for as a comma separated string',
                        type=str, default='all')
    args = parser.parse_args(arguments)

    if not os.path.isdir(args.data_dir):
        os.makedirs(args.data_dir)
    tasks = get_tasks(args.tasks)

    for task in tasks:
        download_and_extract(task, args.data_dir)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))