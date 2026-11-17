"""Publish verified JSON/figures from a dedicated clean clone; never stage raw arrays.

Does not touch the research checkout's index or running solver code. Uses the
user's existing git credential helper, never reads or prints credentials.
"""
raise SystemExit('Retired publisher: use Hyperspectral_HYU/tools/public_results.py; previous remote is prohibited.')
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np
import batch_runner as b

HERE=Path(__file__).resolve().parent;TRACK=HERE.parent
REMOTE='https://github.com/OMS-HYU/capstone_optics.git'
DEST=Path('싱글_하이퍼스펙/결과/자동게시_20261117')


def collect():
    summary=TRACK/'결과/all_photos_660_float64_20261117/results.json'
    records=[]
    if summary.exists():
        for ev in b.read_json(summary)['results']:
            run=TRACK/'실험/all_photos_660_float64_20261117'/ev['scene']
            phase=run/'recovery_float64'/f"phase_{ev['lbfgs_accepted']:03}.npy"
            records.append((ev,run,phase,run/'recovery_float64/comparison.png'))
    tank=TRACK/'실험/tank_fresh_672912_20261117/tank'
    if (tank/'recovery_float64/evaluation.json').exists():
        original=b.read_json(tank/'recovery_float64/evaluation.json');m=original['results']['float64_L60']
        ev=dict(scene='tank',wavelength_nm=660,gd_accepted=150,lbfgs_accepted=60,
                circular_rmse_rad=m['circular_rmse_rad'],linear_rmse_rad=m['linear_rmse_rad'],
                reference_float32_sensor_l2=m['reference_float32_sensor_l2'],
                recovery_float64_sensor_l2=m['recovery_float64_sensor_l2'],quality_pass=original['quality_pass'],
                sensor_sha256=original['sensor_sha256'],truth_sha256=original['truth_sha256'],
                phase_sha256=m['phase_sha256'],variant=original['variant'],
                truth_used_for_evaluation_only=True,preview='comparison.png')
        records.append((ev,tank,tank/'recovery_float64/phase_060.npy',None))
    for ev,run,phase,preview in records:
        man=b.read_json(run/'manifest.json')
        for name,path in [('phase_sha256',phase),('sensor_sha256',run/'sensor.npy'),('truth_sha256',run/'private_truth.npy')]:
            if b.array_sha(path)!=ev[name]:raise ValueError(f"Unverified {ev['scene']} {name}")
        if man['geometry']!=[5713,2857,512,3000] or man['observation']!='amplitude':raise ValueError('Unexpected physics')
        if not np.isfinite([ev['circular_rmse_rad'],ev['reference_float32_sensor_l2']]).all():raise ValueError('Nonfinite metric')
    return sorted(records,key=lambda x:x[0]['scene'])


def prepare(root):
    records=collect();payload=root/'payload';payload.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(root/'matplotlib_cache')
    from render_five_panel import render
    published=[]
    for source,run,phase,preview in records:
        ev=dict(source);ev['preview']=f"{ev['scene']}/comparison.png"
        dest=payload/ev['scene'];dest.mkdir(exist_ok=True)
        if preview is None:
            ev['figure']=render(run,dest/'comparison.png',phase_path=phase)
        else:shutil.copy2(preview,dest/'comparison.png')
        b.write_json(dest/'evaluation.json',ev)
        published.append(ev)
    result=dict(schema_version=1,scope='Verified native5713 amplitude results; float64 refinement variants separate from four legacy baseline results',
                results=published,baseline_reference='../최신_20261116/native_results.json',
                physics_authority='Latest GPT 싱글스펙 revision',raw_arrays_in_git=False)
    b.write_json(payload/'results.json',result)
    rows=['# 검증된 로컬 싱글 결과 자동 게시','',
          'native5713² amplitude 관측. 정밀도 변형은 기존 공통 float32 기준과 구분한다. 원본 배열·optimizer는 로컬 보존.',
          '피겨: 원본 / 관측 스페클(amplitude) / 실제 위상 / 추정 위상 / wrapped 오차. 물리조건은 GPT 싱글스펙 최신 개정 기준.',
          '', '| 장면 | GD / 정밀 횟수 | Circular RMSE(rad) | Sensor L2 | 판정 | 피겨 |',
          '|---|---:|---:|---:|---|---|']
    for ev in published:
        scene=ev['scene'];rows.append(f"| {scene} | {ev['gd_accepted']} / {ev['lbfgs_accepted']} | {ev['circular_rmse_rad']:.7f} | {100*ev['reference_float32_sensor_l2']:.5f}% | {'통과' if ev['quality_pass'] else '기준 미달'} | [보기]({scene}/comparison.png) |")
    rows+=['','각 평가 JSON에 실제 위상/센서/정답 SHA-256과 횟수가 있다. 조기 종료 시 실제 accepted count를 표시한다.',
           '이 Git 반영과 별도로 GitHub Pages 배포 상태를 확인해야 한다. 기존 대화형 사이트는 변경하지 않는다.','']
    (payload/'README.md').write_text('\n'.join(rows),encoding='utf-8')
    files=[p for p in payload.rglob('*') if p.is_file()]
    if any(p.suffix not in ('.json','.png','.md') for p in files):raise ValueError('Unexpected publication artifact')
    digest=hashlib.sha256()
    for p in sorted(files):digest.update(str(p.relative_to(payload)).encode());digest.update(p.read_bytes())
    return payload,[str(p.relative_to(payload)) for p in files],digest.hexdigest(),len(published)


def git(args,clone):
    proc=subprocess.run(['git',*args],cwd=clone,text=True,encoding='utf-8',errors='replace',capture_output=True)
    if proc.returncode:raise RuntimeError(f"git {args[0]} failed: {proc.stderr[-1500:]}")
    return proc.stdout.strip()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--watch',action='store_true');a=ap.parse_args()
    root=TRACK/'실험/git_result_publisher_20261117';root.mkdir(parents=True,exist_ok=True)
    with b.batch_lock(root/'publisher.lock'):
        while True:
            try:
                payload,files,digest,count=prepare(root)
                if a.prepare_only:
                    print(json.dumps(dict(verified_results=count,files=files,digest=digest,remote=REMOTE),ensure_ascii=False),flush=True);return
                statepath=root/'status.json';previous=b.read_json(statepath) if statepath.exists() else {}
                if previous.get('pushed_digest')!=digest:
                    clone=root/'checkout'
                    if not (clone/'.git').exists():
                        subprocess.run(['git','clone','--single-branch','--branch','main',REMOTE,str(clone)],check=True)
                    if git(['remote','get-url','origin'],clone)!=REMOTE:raise ValueError('Unexpected remote')
                    git(['pull','--ff-only','origin','main'],clone)
                    for relative in files:
                        target=clone/DEST/relative;target.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copy2(payload/relative,target)
                    paths=[str(DEST/relative).replace('\\','/') for relative in files]
                    git(['add','-f','--',*paths],clone)
                    diff=git(['-c','core.quotepath=false','diff','--cached','--name-only'],clone)
                    if not set(diff.splitlines()).issubset(set(paths)):
                        raise RuntimeError('Unexpected staged files in publication clone; refusing commit')
                    if diff:
                        git(['-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-m',f'Publish {count} verified native single reconstructions and five-panel figures'],clone)
                    head=git(['rev-parse','HEAD'],clone)
                    git(['push','origin','HEAD:main'],clone)
                    remote_head=git(['ls-remote','origin','refs/heads/main'],clone).split()[0]
                    if head!=remote_head:raise RuntimeError('Remote head verification failed')
                    b.write_json(statepath,dict(status='pushed',verified_results=count,commit=head,pushed_digest=digest,
                                 remote=REMOTE,files=paths,pages_deployment='not_verified',updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
                    print('PUSHED',count,head,flush=True)
                if not a.watch:return
            except Exception as exc:
                b.write_json(root/'last_error.json',dict(error=str(exc),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
                print('PUBLISH_ERROR',str(exc),flush=True)
                if not a.watch:raise
            if (root/'STOP').exists():return
            time.sleep(30)


if __name__=='__main__':main()
