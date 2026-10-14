# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
import tempfile
from pathlib import Path
import sm


def main():
    r = sm.job_request("asr_check", "stages/transcribe/R1", "stages/cut/R0", ["--x", "1"], run="20261014T070509Z-abc1234",
                       sha="abc1234", instance="ml.g5.xlarge", spot=False)
    assert r["AlgorithmSpecification"]["TrainingImage"] == sm.IMAGE_GPU
    assert r["RoleArn"] == sm.ROLE and r["ResourceConfig"]["InstanceType"] == "ml.g5.xlarge"
    hp = r["HyperParameters"]
    assert hp["sagemaker_program"] == "pipeline/sm_entry.py"
    assert hp["sagemaker_submit_directory"] == "s3://sarvam-parta-114872709824/code/abc1234.tar.gz"
    assert hp["stage"] == "asr_check" and hp["run"] == "20261014T070509Z-abc1234" and all(isinstance(v, str) for v in hp.values())
    assert r["CheckpointConfig"] == {"S3Uri": "s3://sarvam-parta-114872709824/stages/asr_check/20261014T070509Z-abc1234/_ckpt/",
                                     "LocalPath": "/opt/ml/checkpoints"}
    assert len(r["TrainingJobName"]) <= 63 and "EnableManagedSpotTraining" not in r
    s = sm.job_request("asr_check", "stages/transcribe/R1", "stages/cut/R0", [], run="R", sha="abc1234", instance="ml.g4dn.xlarge", spot=True)
    assert s["EnableManagedSpotTraining"] is True and s["StoppingCondition"]["MaxWaitTimeInSeconds"] >= s["StoppingCondition"]["MaxRuntimeInSeconds"]
    c = sm.job_request("audio_qc", "stages/cut/R0", "stages/cut/R0", [], run="R", sha="abc1234", instance="ml.c5.xlarge", spot=False)
    assert c["AlgorithmSpecification"]["TrainingImage"] == sm.IMAGE_CPU

    # stage_deps: what sm_entry installs into the image's python; torch stays the image's (it matches its CUDA)
    root = Path(sm.__file__).resolve().parent.parent
    assert sm.stage_deps(root / "validate.py") == [] and sm.stage_deps(root / "pipeline/test_s3.py") == ["boto3"]
    p = Path(tempfile.mkdtemp()) / "stage.py"
    p.write_text('# /// script\n# requires-python = ">=3.12"\n# dependencies = [\n#   "numpy>=2",\n#   "torch==2.9",\n'
                 '#   "torchaudio",\n#   "torchmetrics",\n# ]\n# ///\nimport numpy\n')
    assert sm.stage_deps(p) == ["numpy>=2", "torchmetrics"]

    # dirty_tree: submit refuses uncommitted code
    sm._dirty = lambda: True
    assert sm.submit("audio_qc", "stages/cut/R0", "stages/cut/R0", [], run="R", instance="ml.c5.xlarge", spot=False,
                     sagemaker=object(), s3=object()) == 3
    # account instance limits: ResourceLimitExceeded is waited out, other errors raise at once
    from botocore.exceptions import ClientError
    class SM:
        def __init__(self, fails, code="ResourceLimitExceeded"):
            self.fails, self.code, self.calls = fails, code, 0
        def create_training_job(self, **kw):
            self.calls += 1
            if self.calls <= self.fails:
                raise ClientError({"Error": {"Code": self.code}}, "CreateTrainingJob")
            return {"TrainingJobArn": "arn"}
    naps = []
    s = SM(2)
    assert sm._create(s, {}, wait=5, tries=4, sleep=naps.append) == {"TrainingJobArn": "arn"} and s.calls == 3 and naps == [5, 5]
    for bad in (SM(9), SM(1, "ValidationException")):
        try:
            sm._create(bad, {}, wait=5, tries=4, sleep=naps.append)
            raise AssertionError("should have raised")
        except ClientError:
            pass
    print("ok: sm")


if __name__ == "__main__":
    main()
