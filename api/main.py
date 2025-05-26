import shutil
import logging
from pathlib import Path
import subprocess as sp
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


@app.post("/biology/ipd/rfdiffusion/generate")
async def generate(
    pdb_file: UploadFile = File(...),
    output_prefix: str = Form("rfd_output"),
    contigmap: str = Form("[]"),
    num_designs: int = Form(40),
    diffuser_partial_T: Union[int, float] = Form(None),
):
    # check that output_prefix is not a path
    if Path(output_prefix).parent != Path("."):
        raise HTTPException(status_code=400, detail="output_prefix must not be a path")

    # read the pdb file, if it exists
    if not pdb_file.filename.endswith(".pdb"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .pdb files are accepted.")
    if not pdb_file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    # Save the uploaded file to a temporary location
    input_file_path = BASE_DIR / Path("temp.pdb")
    input_file_path.write_text(pdb_file.file.read().decode("utf-8").strip())

    # make the output directory if it doesn't exist
    output_dir = BASE_DIR / Path("rfdiffusion")
    if output_dir.exists():
        logging.warning(f"Removing old output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)

    # build the command
    rfdiffusion_cmd = [
        "python3.9",
        "/app/RFdiffusion/scripts/run_inference.py",
        f"inference.input_pdb={input_file_path}",
        f"inference.output_prefix={output_dir / Path(output_prefix)}",
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
        input_file_path.unlink()
    except FileNotFoundError:
        logging.warning(f"Temporary pdb file not found: {input_file_path}")
    except Exception as e:
        logging.error(f"Error removing temporary pdb file: {e}")

    return {
        "status": "success",
        "output_dir": str(output_dir),
        "elapsed_time": elapsed_time,
    }
