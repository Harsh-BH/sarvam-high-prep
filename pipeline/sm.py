"""Submit a stage as a SageMaker training job (design §4.1). The job runs run.run_local inside the container."""
import json, re, subprocess, sys, time, tomllib
import s3 as store
from run import ROOT, git_sha

IMAGE_GPU = "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker-v2.2-2026-09-07-22-13-38"
IMAGE_CPU = "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training:2.9.0-cpu-py312-ubuntu22.04-sagemaker-v2.2-2026-09-07-22-13-44"
ROLE = "arn:aws:iam::114872709824:role/sarvam-sm-role"
GPU_FAMILIES = ("ml.g", "ml.p")
IMAGE_OWNS = {"torch", "torchaudio", "torchvision"}  # built against the image's CUDA; never reinstalled


def stage_deps(script):
    """PEP 723 `dependencies` of a stage script, minus what the image owns."""
    m = re.search(r"(?m)^# /// script$\s(?P<content>(^#(| .*)$\s)+)^# ///$", script.read_text())
    if not m:
        return []
    toml = "".join(l[2:] if l.startswith("# ") else l[1:] for l in m.group("content").splitlines(keepends=True))
    return [d for d in tomllib.loads(toml).get("dependencies", []) if re.match(r"[A-Za-z0-9._-]+", d).group().lower() not in IMAGE_OWNS]


def _dirty():
    return bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip())


def job_request(stage, src, audio_src, params, *, run, sha, instance, spot):
    b = store.BUCKET
    # ponytail: one job name per run id, so re-submitting a failed --sm run under the same --run is refused by
    # SageMaker (names are unique); add an attempt suffix when resuming on SageMaker is needed
    name = re.sub(r"[^A-Za-z0-9-]", "-", f"sarvam-{stage}-{run}")[:63].rstrip("-")
    req = {
        "TrainingJobName": name,
        "AlgorithmSpecification": {"TrainingImage": IMAGE_GPU if instance.startswith(GPU_FAMILIES) else IMAGE_CPU,
                                   "TrainingInputMode": "File"},
        "RoleArn": ROLE,
        "HyperParameters": {"sagemaker_program": "pipeline/sm_entry.py",
                            "sagemaker_submit_directory": f"s3://{b}/code/{sha}.tar.gz",
                            "stage": stage, "src": src, "audio_src": audio_src, "run": run, "params": json.dumps(params)},
        "OutputDataConfig": {"S3OutputPath": f"s3://{b}/stages/_sm_output/"},
        "ResourceConfig": {"InstanceType": instance, "InstanceCount": 1, "VolumeSizeInGB": 100},
        "StoppingCondition": {"MaxRuntimeInSeconds": 6 * 3600},
        "CheckpointConfig": {"S3Uri": f"s3://{b}/stages/{stage}/{run}/_ckpt/", "LocalPath": "/opt/ml/checkpoints"},
        "Environment": {"AWS_DEFAULT_REGION": store.REGION},
        "Tags": [{"Key": "project", "Value": "sarvam-parta"}],
    }
    if spot:
        req["EnableManagedSpotTraining"] = True
        req["StoppingCondition"]["MaxWaitTimeInSeconds"] = 12 * 3600
    return req


def submit(stage, src, audio_src, params, *, run, instance, spot, sagemaker=None, s3=None):
    if _dirty():
        print("refusing: commit first, a job must map to one git sha", file=sys.stderr)
        return 3
    sha = git_sha()
    tar = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar.gz", "HEAD"], capture_output=True, check=True).stdout
    store.put_once(f"code/{sha}.tar.gz", tar, s3)
    if sagemaker is None:
        sagemaker = store.session().client("sagemaker")
    req = job_request(stage, src, audio_src, params, run=run, sha=sha, instance=instance, spot=spot)
    sagemaker.create_training_job(**req)
    print(f"submitted {req['TrainingJobName']}; logs: /aws/sagemaker/TrainingJobs/{req['TrainingJobName']}", file=sys.stderr)
    while True:
        d = sagemaker.describe_training_job(TrainingJobName=req["TrainingJobName"])
        if d["TrainingJobStatus"] in ("Completed", "Failed", "Stopped"):
            print(d["TrainingJobStatus"], d.get("FailureReason", ""), file=sys.stderr)
            return 0 if d["TrainingJobStatus"] == "Completed" else 1
        time.sleep(30)
