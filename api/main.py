import shutil
import logging
from pathlib import Path
import subprocess as sp
from tempfile import NamedTemporaryFile
import time
from typing import Union

from fastapi import FastAPI, File, Form, UploadFile, HTTPException


BASE_DIR = Path("/workspace")
SERVICE_CACHE_DIR = Path("/cache")

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="RFdiffusion API")


@app.get("/v1/health/ready")
async def health_check():
    return {"status": "ready"}


@app.post("/api/generate")
async def generate(
    pdb_file: UploadFile = File(...),
    contigmap: str = Form("[]"),
    num_designs: int = Form(40),
    diffuser_partial_T: Union[int, float] = Form(None),
    run_output_dirname: str = Form(""),
):
    # read the pdb file, if it exists
    if not pdb_file.filename.endswith(".pdb"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .pdb files are accepted.")
    if not pdb_file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    # Save the uploaded file to a temporary location
    tmp_infile = NamedTemporaryFile(suffix=".pdb", delete=True)
    tmp_infile.write(pdb_file.file.read())
    tmp_infile.flush()

    # make the output directory if it doesn't exist
    output_dir = BASE_DIR / run_output_dirname
    output_dir.mkdir(exist_ok=True)

    # build the command
    rfdiffusion_cmd = [
        "python3.9",
        "/app/RFdiffusion/scripts/run_inference.py",
        f"inference.input_pdb={Path(tmp_infile.name)}",
        f"inference.output_prefix={output_dir / 'design'}",
        f"inference.model_directory_path={SERVICE_CACHE_DIR}",
        f"inference.num_designs={num_designs}",
        f"contigmap.contigs={contigmap}",
    ]
    if diffuser_partial_T is not None:
        rfdiffusion_cmd.append(f"diffuser.partial_T={diffuser_partial_T}")

    # run the command
    start_time = time.time()
    try:
        logging.info(f"Running RFdiffusion with command: {' '.join(rfdiffusion_cmd)}")
        _ = sp.run(rfdiffusion_cmd)
    except sp.CalledProcessError as e:
        logging.error(f"Command failed with error: {e.stderr}")
        raise HTTPException(status_code=500, detail="Command failed")
    elapsed_time = time.time() - start_time

    # remove the temporary pdb file
    try:
        tmp_infile.close()
        logging.info(f"Temporary pdb file removed: {tmp_infile}")
    except FileNotFoundError:
        logging.warning(f"Temporary pdb file not found: {tmp_infile}")
    except Exception as e:
        logging.error(f"Error removing temporary pdb file: {e}")

    return {
        "status": "success",
        "output_dir": str(output_dir),
        "elapsed_time": elapsed_time,
    }
