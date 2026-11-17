"""Public-only Hyperspectral_HYU publication, from this project's native workspace.

Preserves the existing 39-scene website; publishes locally verified records,
figures and machine-readable public JSON without private image dependencies.
"""
import argparse
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];TRACK=ROOT/'싱글/native';CODE=TRACK/'코드'
sys.path.insert(0,str(CODE))
import batch_runner as b
REMOTE='https://github.com/OMS-HYU/Hyperspectral_HYU.git'
PUBLIC='https://oms-hyu.github.io/Hyperspectral_HYU/'
CONTROL=ROOT/'실험/공개게시_20261117'


def sha(path):return b.sha256_file(path)


def records():
    output=[]
    baseline=b.read_json(TRACK/'결과/최신_20261116/native_results.json')
    for old in baseline['ordinary']:
        scene=old['id'];source=TRACK/'결과/최신_20261116'/old['preview']
        ev=dict(scene=scene,name=old['scene'],wavelength_nm=old['wavelength_nm'],
            gd_accepted=old['gd_accepted'],lbfgs_accepted=old['lbfgs_accepted'],
            extra_lbfgs_accepted=old.get('extra_lbfgs_accepted',0),
            circular_rmse_rad=old['final_circular_rmse_rad'],linear_rmse_rad=old['final_linear_rmse_rad'],
            sensor_l2=old['sensor_l2'],quality_pass=old['final_circular_rmse_rad']<=.15 and old['sensor_l2']<=.025,
            phase_sha256=old['phase_sha256'],sensor_sha256=old['sensor_sha256'],truth_sha256=old['truth_sha256'],
            variant=old['final_method'],verification='Previously verified baseline; raw arrays may be in historical archives',
            figure_stage=old.get('preview_stage','historical preview'),five_panel=False)
        output.append((ev,source,None))
    index=TRACK/'결과/all_photos_660_float64_20261117/results.json'
    if index.exists():
        for old in b.read_json(index)['results']:
            run=TRACK/'실험/all_photos_660_float64_20261117'/old['scene']
            phase=run/'recovery_float64'/f"phase_{old['lbfgs_accepted']:03}.npy"
            for key,path in [('phase_sha256',phase),('sensor_sha256',run/'sensor.npy'),('truth_sha256',run/'private_truth.npy')]:
                if b.array_sha(path)!=old[key]:raise ValueError(f"Unverified {old['scene']} {key}")
            ev=dict(old,sensor_l2=old['reference_float32_sensor_l2'],five_panel=True,
                    verification='Phase/sensor/truth SHA-256 verified in this public project workspace')
            output.append((ev,run/'recovery_float64/comparison.png',None))
    tank=TRACK/'실험/tank_fresh_672912_20261117/tank'
    if (tank/'recovery_float64/evaluation.json').exists():
        old=b.read_json(tank/'recovery_float64/evaluation.json');m=old['results']['float64_L60']
        phase=tank/'recovery_float64/phase_060.npy'
        for key,path in [('phase_sha256',phase),('sensor_sha256',tank/'sensor.npy'),('truth_sha256',tank/'private_truth.npy')]:
            expected=m[key] if key=='phase_sha256' else old[key]
            if b.array_sha(path)!=expected:raise ValueError('Tank array hash mismatch')
        ev=dict(scene='tank',name='Tank',wavelength_nm=660,gd_accepted=150,lbfgs_accepted=60,
                circular_rmse_rad=m['circular_rmse_rad'],linear_rmse_rad=m['linear_rmse_rad'],
                sensor_l2=m['reference_float32_sensor_l2'],quality_pass=old['quality_pass'],
                phase_sha256=m['phase_sha256'],sensor_sha256=old['sensor_sha256'],truth_sha256=old['truth_sha256'],
                variant=old['variant'],five_panel=True,verification='Phase/sensor/truth SHA-256 verified',
                reference_float32_lbfgs_accepted=37)
        output.append((ev,None,tank))
    return sorted(output,key=lambda x:x[0]['scene'])


def prepare():
    CONTROL.mkdir(parents=True,exist_ok=True);os.environ['MPLCONFIGDIR']=str(CONTROL/'matplotlib_cache')
    from render_five_panel import render
    destination=ROOT/'공개결과/싱글';destination.mkdir(parents=True,exist_ok=True)
    assets=ROOT/'site/assets/native_20261117';assets.mkdir(parents=True,exist_ok=True)
    published=[];paths=[]
    for ev,source,tank in records():
        scene=ev['scene'];case=destination/scene;case.mkdir(exist_ok=True)
        suffix=source.suffix if source else '.png';asset=assets/(scene+suffix)
        if tank:render(tank,asset,phase_path=tank/'recovery_float64/phase_060.npy')
        else:shutil.copy2(source,asset)
        shutil.copy2(asset,case/('comparison'+suffix))
        ev['figure']=PUBLIC+asset.relative_to(ROOT/'site').as_posix()
        ev['evaluation_url']=PUBLIC+f'data/single/{scene}.json'
        ev['public_repository']='OMS-HYU/Hyperspectral_HYU'
        b.write_json(case/'evaluation.json',ev)
        datafile=ROOT/f'site/data/single/{scene}.json';datafile.parent.mkdir(parents=True,exist_ok=True);b.write_json(datafile,ev)
        paths.extend([asset,case/('comparison'+suffix),case/'evaluation.json',datafile]);published.append(ev)
    aggregate=dict(schema_version=1,public_url=PUBLIC,repository=REMOTE,
       physics='Native5713 amplitude; propagation5713/meta2857/object512/sensor3000; fixed background; 400um/6.3mm; sensor1.85um',
       phase_metric='Circular RMSE without piston alignment; linear RMSE reported separately',
       verified_completed_count=sum(x['quality_pass'] for x in published),results=published,
       note='Historical baseline preview stages are labeled. Float64 variants and actual iteration counts remain explicit.')
    for p in [destination/'results.json',ROOT/'site/data/latest_single.json']:
        p.parent.mkdir(parents=True,exist_ok=True);b.write_json(p,aggregate);paths.append(p)
    s=(ROOT/'site/index.html').read_text(encoding='utf-8')
    match=re.search(r'<script[^>]*id=[\'"]dataset[\'"][^>]*>(.*?)</script>',s,re.S)
    if not match:raise ValueError('Existing 39-scene dataset not found')
    dataset=json.loads(match[1]);lookup={x['id']:x for x in dataset}
    if len(dataset)!=39:raise ValueError('Preserve all 39 existing scene selectors')
    for ev in published:
        scene=ev['scene'];x=lookup[scene]
        method=f"native GD{ev['gd_accepted']} + L{ev['lbfgs_accepted']}"
        if ev.get('extra_lbfgs_accepted'):method+=f" + Fourier{ev['extra_lbfgs_accepted']}"
        verdict='통과' if ev['quality_pass'] else '기준 미달'
        detail='native5713 amplitude · '+ev['variant']+' · '+ev['verification']
        latest=dict(w=ev['wavelength_nm'],r=ev['circular_rmse_rad'],s=ev['sensor_l2']*100,
                    method=method,noise=1,verdict=verdict,detail=detail,publication='native_20261117')
        x['latest']=latest;x['native_current']=ev;x['status']=verdict
        x['known']=[v for v in x.get('known',[]) if v.get('publication')!='native_20261117']+[latest]
        title=f"{ev['wavelength_nm']} nm · {method} · RMSE {ev['circular_rmse_rad']:.6f} rad · {verdict}"
        if not ev['five_panel']:title+=' · 보존된 '+ev['figure_stage']
        x['figures']=[dict(mode='known',title=title,src=ev['figure'].replace(PUBLIC,''),publication='native_20261117')]+[
            f for f in x.get('figures',[]) if f.get('publication')!='native_20261117']
        row_pattern=r'<tr\b[^>]*data-id=[\'"]'+re.escape(scene)+r'[\'"][^>]*>.*?</tr>'
        row=re.search(row_pattern,s,re.S)
        if row:
            cells=re.findall(r'<td\b[^>]*>.*?</td>',row[0],re.S)
            if len(cells)>=4:
                cells[1]='<td>'+html.escape(method)+'</td>'
                cells[3]=f'<td class="knowncol"><span class="pill {"ok" if ev["quality_pass"] else "warn"}">{verdict}</span><div class="mini"><b>{ev["circular_rmse_rad"]:.6f} rad</b></div><small>{ev["sensor_l2"]*100:.5f}% · {html.escape(method)}</small><div class="subnote"><a href="data/single/{scene}.json">공개 평가 JSON</a></div></td>'
                position=iter(cells);newrow=re.sub(r'<td\b[^>]*>.*?</td>',lambda m:next(position),row[0],flags=re.S)
                newrow=re.sub(r'(<span class="pill [^"]*knownbadge">).*?(</span>)',lambda m:m[1]+'Known native '+verdict+m[2],newrow)
                s=s[:row.start()]+newrow+s[row.end():]
    # Row edits alter offsets, so locate the dataset again.
    match=re.search(r'<script[^>]*id=[\'"]dataset[\'"][^>]*>(.*?)</script>',s,re.S)
    encoded=json.dumps(dataset,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    s=s[:match.start(1)]+encoded+s[match.end(1):]
    s=re.sub(r'<p class="muted" id="native-public-status">.*?</p>','',s,flags=re.S)
    status=f'<p class="muted" id="native-public-status">최신 native 완료 {aggregate["verified_completed_count"]}개 · <a href="data/latest_single.json">공개 결과 JSON</a> · 피겨/평가 파일 로그인 없이 열람</p>'
    s=s.replace('</h1>','</h1>'+status,1)
    footer=r'<footer>.*?</footer>'
    s=re.sub(footer,'<footer>B17 K=1 · 최신 native5713² amplitude 결과 · 과거3800² 결과는 별도 이력 · 물리조건은 싱글스펙 최신 개정 기준</footer>',s,flags=re.S)
    (ROOT/'site/index.html').write_text(s,encoding='utf-8');paths.append(ROOT/'site/index.html')
    lines=['# 공개 싱글 결과','',f'공개 결과창: {PUBLIC}',f'기계 판독 JSON: {PUBLIC}data/latest_single.json','',
           '| 장면 | GD / 정밀 | Circular RMSE(rad) | Sensor L2 | 피겨 |','|---|---:|---:|---:|---|']
    for ev in published:
        suffix=Path(ev['figure']).suffix
        lines.append(f"| {ev['scene']} | {ev['gd_accepted']} / {ev['lbfgs_accepted']} | {ev['circular_rmse_rad']:.7f} | {100*ev['sensor_l2']:.5f}% | [보기]({ev['scene']}/comparison{suffix}) |")
    lines+=['','각 평가 JSON은 실제 횟수·hash·정밀도변형·이전 기준 그림 단계를 명시한다. 큰 배열/optimizer는 이 공개 프로젝트의 로컬 실험 폴더에 보존한다.','']
    (destination/'README.md').write_text('\n'.join(lines),encoding='utf-8');paths.append(destination/'README.md')
    digest=hashlib.sha256()
    for ev in published:digest.update(json.dumps(ev,sort_keys=True,ensure_ascii=False).encode())
    return paths,digest.hexdigest(),aggregate


def git(args,cwd=ROOT):
    result=subprocess.run(['git',*args],cwd=cwd,capture_output=True,text=True,encoding='utf-8',errors='replace')
    if result.returncode:raise RuntimeError(f'git {args[0]}: {result.stderr[-1600:]}')
    return result.stdout.strip()


def initial_paths():
    files=[]
    for name in ['.gitignore','AGENTS.md','README.md']:
        p=ROOT/name
        if p.exists():files.append(p)
    for directory in ['docs','tools','싱글/native/코드','싱글/native/연구기록','싱글/native/데이터/입력이미지']:
        for p in (ROOT/directory).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.ps1','.md','.json','.png','.jpg']:files.append(p)
    for name in ['AGENTS.md','README.md','run.py','requirements-native.txt','requirements.txt','데이터/README.md']:
        p=TRACK/name
        if p.exists():files.append(p)
    files += [p for p in (ROOT/'site').rglob('*') if p.is_file() and p.suffix in ['.html','.json','.png','.jpg','.webp']]
    return files


def commit_files(paths,message,cwd=ROOT):
    relative=sorted({p.relative_to(cwd).as_posix() for p in paths})
    if any((cwd/p).stat().st_size>90*1024*1024 for p in relative):raise ValueError('Large raw artifact in publication set')
    git(['add','--',*relative],cwd)
    staged=git(['-c','core.quotepath=false','diff','--cached','--name-only'],cwd).splitlines()
    if not set(staged).issubset(set(relative)):raise ValueError('Unexpected pre-existing staged changes; refusing commit')
    if staged:git(['-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-m',message],cwd)
    return git(['rev-parse','HEAD'],cwd)


def publish(paths,count,first):
    if git(['remote','get-url','origin'])!=REMOTE:raise ValueError('Only the public Hyperspectral_HYU remote is allowed')
    if git(['branch','--show-current'])!='main':raise ValueError('Expected public project main branch')
    git(['fetch','origin','main','gh-pages'])
    ahead,behind=map(int,git(['rev-list','--left-right','--count','HEAD...origin/main']).split())
    if behind:raise RuntimeError('Public main advanced; integrate it before publication without overwriting local work')
    main=commit_files(paths+(initial_paths() if first else []),f'Migrate native single workspace and publish {count} verified results' if first else f'Update public single results: {count} verified')
    git(['push','origin','main'])
    if git(['ls-remote','origin','refs/heads/main']).split()[0]!=main:raise RuntimeError('Public main push verification failed')
    pages=CONTROL/'gh-pages'
    if not (pages/'.git').exists():git(['worktree','add','--detach',str(pages),'origin/gh-pages'])
    else:git(['merge','--ff-only','origin/gh-pages'],pages)
    pagefiles=[]
    for source in (ROOT/'site').rglob('*'):
        if not source.is_file() or source.suffix not in ['.html','.json','.png','.jpg','.webp']:continue
        target=pages/source.relative_to(ROOT/'site');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);pagefiles.append(target)
    pagecommit=commit_files(pagefiles,f'Publish native single results: {count} verified, public figures and JSON',pages)
    git(['push','origin','HEAD:gh-pages'],pages)
    if git(['ls-remote','origin','refs/heads/gh-pages']).split()[0]!=pagecommit:raise RuntimeError('Public gh-pages push verification failed')
    return main,pagecommit


def pages_matches(expected):
    try:
        request=urllib.request.Request(PUBLIC+'data/latest_single.json',headers={'Cache-Control':'no-cache','User-Agent':'Hyperspectral-HYU-results-check'})
        with urllib.request.urlopen(request,timeout=20) as response:actual=json.load(response)
        return actual==expected
    except Exception:return False


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--watch',action='store_true');a=ap.parse_args()
    CONTROL.mkdir(parents=True,exist_ok=True)
    with b.batch_lock(CONTROL/'publisher.lock'):
        while True:
            try:
                paths,digest,result=prepare();statepath=CONTROL/'status.json';state=b.read_json(statepath) if statepath.exists() else {}
                count=result['verified_completed_count']
                if a.prepare_only:
                    print(json.dumps(dict(verified_completed=count,source_files=len(initial_paths()),public_url=PUBLIC,
                        files=[p.relative_to(ROOT).as_posix() for p in paths],remote=REMOTE),ensure_ascii=True),flush=True);return
                if state.get('digest')!=digest:
                    main_sha,page_sha=publish(paths,count,first=not state.get('main_commit'))
                    state=dict(status='pushed_pages_pending',digest=digest,verified_completed=count,main_commit=main_sha,
                               pages_commit=page_sha,repository=REMOTE,public_url=PUBLIC,source_workspace=str(ROOT),pid=os.getpid())
                    b.write_json(statepath,state);print('PUBLIC_PUSHED',count,main_sha,page_sha,flush=True)
                if state.get('status')!='published' and pages_matches(result):
                    state['status']='published';b.write_json(statepath,state);print('PUBLIC_PAGES_VERIFIED',count,flush=True)
                if not a.watch:return
            except Exception as exc:
                b.write_json(CONTROL/'last_error.json',dict(error=str(exc),time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
                print('PUBLICATION_ERROR',str(exc),flush=True)
                if not a.watch:raise
            if (CONTROL/'STOP').exists():return
            time.sleep(30)


if __name__=='__main__':main()
