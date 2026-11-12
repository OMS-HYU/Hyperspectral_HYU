"""Full-grid recommendation benchmark: one independent case per process.
No optimizer changes, no reduced grids, no truth supplied to inverse.
"""
from pathlib import Path
import argparse, csv, hashlib, json, os, shutil, subprocess, sys, time, traceback

ROOT=Path(__file__).resolve().parents[2]
RBC=("normal","variable","tilt","double","triple")
GENERAL=("checkerboard","boat","tank","pepper","lena","barbara","house","starfish","monarch","airplane","parrot","man","couple","camera_set12","pepper_set12","boat_set12","mandrill","aerial","bark","brick_wall","electron_micrograph","motioncar","plane_c4l","random_texture","sand","stream","boat_c4l","tank_c4l")
SCENES=RBC+GENERAL

def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")

def one(scene,dest):
    method="spectral" if scene in RBC else "pyramid"
    dest.mkdir(parents=True,exist_ok=False)
    run=dest/"work"
    log=dest/"run.log"
    result=dict(scene=scene,method=method,wavelength_nm=660,mode="known",
                noise_seed=672906,geometry_seed=20261127,noise_pct_mean=1,
                geometry=[3800,1429,512,3000],git_sha=os.getenv("GITHUB_SHA"),
                status="error",passed=None)
    t=time.monotonic()
    cmd=[sys.executable,"-u","-m","싱글.run","single","--scene",scene,
         "--true-wavelength","660","--mode","known","--wavelength","660",
         "--method",method,"--seed","672906","--geometry-seed","20261127",
         "--output",str(run.resolve())]
    result["command"]=cmd
    try:
        with log.open("w",encoding="utf-8") as f:
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                               text=True,encoding="utf-8",errors="replace",bufsize=1)
            for line in p.stdout:
                print(line,end="",flush=True)
                f.write(line);f.flush()
            result["returncode"]=p.wait()
        inv=run/"inverse";obs=run/"observation"
        for f in ("evaluation.json","recovery.json","history.json","five_panel.png",
                  "convergence.png","recovered.npz","gd_sealed.npz"):
            if (inv/f).is_file():shutil.copy2(inv/f,dest/f)
        for f in ("generation.json","private_truth.npz"):
            if (obs/f).is_file():shutil.copy2(obs/f,dest/f)
        if result["returncode"]==0 and (dest/"evaluation.json").is_file():
            ev=json.loads((dest/"evaluation.json").read_text(encoding="utf-8"))
            audit=json.loads((dest/"recovery.json").read_text(encoding="utf-8"))
            generation=json.loads((dest/"generation.json").read_text(encoding="utf-8"))
            assert audit["geometry"]==result["geometry"]
            assert audit["method"]==method and not audit["truth_in_inverse"]
            assert audit["sensor_sha256"]==generation["sensor_sha256"]
            assert ev["actual_wavelength_nm"]==ev["estimated_wavelength_nm"]==660
            result.update(status="finished",passed=ev["passed"],evaluation=ev,
                recovery=audit, generation=generation)
        else:
            result["error"]="CLI failed or evaluation missing; see run.log"
    except Exception:
        result["error"]=traceback.format_exc()
        print(result["error"],flush=True)
    finally:
        result["wall_seconds"]=time.monotonic()-t
        write_json(dest/"case_result.json",result)
        # Sensor and prediction can be regenerated from frozen code, truth, seeds
        # and original B17. Avoid uploading 100+ MB per case as public CI artifacts.
        if run.exists():shutil.rmtree(run)
        print("BENCHMARK_CASE",json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if result["status"]=="finished" else 1

def aggregate(folder,dest):
    dest.mkdir(parents=True,exist_ok=True)
    records={json.loads(p.read_text(encoding="utf-8"))["scene"]:json.loads(p.read_text(encoding="utf-8"))
             for p in folder.rglob("case_result.json")}
    rows=[]
    for scene in SCENES:
        r=records.get(scene,{})
        e=r.get("evaluation",{});a=r.get("recovery",{})
        rows.append(dict(scene=scene,method="spectral" if scene in RBC else "pyramid",
          status=r.get("status","missing"),passed=r.get("passed"),
          full_rmse_rad=e.get("full_rmse_rad"),roi_rmse_rad=e.get("roi_rmse_rad"),
          sensor_l2=e.get("sensor_l2"),gd=e.get("gd_iterations"),lbfgs=e.get("lbfgs_iterations"),
          groups_passed=e.get("groups_passed"),groups_total=e.get("groups_total"),
          layer_3_rmse_rad=e.get("layer_3_rmse_rad"),forward=a.get("forward"),adjoint=a.get("adjoint"),
          seconds=r.get("wall_seconds"),termination=a.get("termination"),sensor_sha256=a.get("sensor_sha256")))
    write_json(dest/"summary.json",rows)
    with (dest/"summary.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    n=sum(r["status"]=="finished" for r in rows);p=sum(r["passed"] is True for r in rows)
    lines=["# 싱글 추천 알고리즘 전체 검증","",
        f"완료 {n}/{len(rows)}, 기준 통과 {p}/{len(rows)}. 660 nm Known, 1% noise, 전체 B17 격자.",
        "", "| Scene | Method | Status | Full RMSE (rad) | ROI (rad) | Sensor L2 (%) | GD+LBFGS | Pass |",
        "|---|---|---|---:|---:|---:|---:|---|"]
    def fmt(x,scale=1):return "-" if x is None else f"{scale*x:.6f}"
    for r in rows:
        lines.append(f"| {r['scene']} | {r['method']} | {r['status']} | {fmt(r['full_rmse_rad'])} | {fmt(r['roi_rmse_rad'])} | {fmt(r['sensor_l2'],100)} | {r['gd'] or '-' }+{r['lbfgs'] or '-'} | {r['passed']} |")
    lines+=["","성공적인 CLI 종료와 수치 통과는 별개. 미완료·오류를 통과로 기록하지 않는다. 다른 파장/Blind/실측 일반화의 근거가 아니다."]
    report="\n".join(lines)+"\n"
    (dest/"REPORT_KO.md").write_text(report,encoding="utf-8")
    print(report)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"],"a",encoding="utf-8") as f:f.write(report)
    return 0 if n==len(rows) else 1

if __name__=="__main__":
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="command",required=True)
    s=sub.add_parser("case");s.add_argument("--scene",choices=SCENES,required=True);s.add_argument("--output",type=Path,required=True)
    a=sub.add_parser("aggregate");a.add_argument("--input",type=Path,required=True);a.add_argument("--output",type=Path,required=True)
    v=p.parse_args()
    sys.exit(one(v.scene,v.output) if v.command=="case" else aggregate(v.input,v.output))
