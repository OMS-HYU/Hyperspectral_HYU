"""All remaining photos under the user's website intensity/figure specification."""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
import batch_runner as b

HERE=Path(__file__).resolve().parent;TRACK=HERE.parent;REPO=TRACK.parent.parent
EXCLUDED={'mandrill','camera_rot90','barbara','boat','tank'}
FILES=[HERE/n for n in ['site_reference_batch.py','site_intensity_run.py','site_intensity_model.py',
                       'site_reference_figure.py','render_five_panel.py','tank_precision_recovery.py','native_cli.py','native_model.py','batch_runner.py']]
FILES += [REPO/'공통'/n for n in ['b17_core.py','b17_loader.py','propagation.py']]


def main():
    raise RuntimeError('Superseded: user clarified website is a figure-only reference. Use latest 싱글스펙 native physics and all_photos_batch.py.')
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ['input-archive','optics-archive','output']:ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args()
    for k in ['input_archive','optics_archive','output']:setattr(a,k,getattr(a,k).resolve())
    a.output.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(a.output/'matplotlib_cache')
    with b.batch_lock(a.output/'batch.lock'):
        if b.sha256_file(a.optics_archive)!=b.EXPECTED_OPTICS:raise ValueError('B17 optics hash mismatch')
        extracted,_,entries=b.inspect_input(a.input_archive,a.output/'_source')
        cases=[(i,x) for i,x in enumerate(entries) if x['id'] not in EXCLUDED]
        cases.sort(key=lambda ix:'RBC' in ix[1].get('name',''))
        identity=dict(input_sha256=b.sha256_file(a.input_archive),optics_sha256=b.EXPECTED_OPTICS,
                      code={str(p.relative_to(REPO)):b.sha256_file(p) for p in FILES},
                      cases=[x['id'] for i,x in cases],wavelength_nm=660,seed_base=674000,
                      geometry=[3800,1429,512,3000],observation='intensity',amplitude_outside=0)
        ip=a.output/'identity.json'
        if ip.exists() and b.read_json(ip)!=identity:raise ValueError('Batch identity changed')
        b.write_json(ip,identity)
        qp=a.output/'queue.json';previous=b.read_json(qp) if qp.exists() else {}
        queue=dict(status='running',pid=os.getpid(),total=len(cases),results=previous.get('results',[]),
                   cases=[dict(id=x['id'],name=x['name'],seed=674000+i) for i,x in cases],
                   excluded_from_remaining_scope=sorted(EXCLUDED),
                   note='Excluded legacy results are not counted as verified under this new intensity protocol',
                   figure_order=['original','observed speckle','true phase','estimated phase','wrapped error'],
                   reference_url='https://oms-hyu.github.io/Hyperspectral_HYU/')
        def save():
            queue['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());b.write_json(qp,queue)
        save()
        b.write_json(a.output/'resume_command.json',dict(python=sys.executable,args=[str(Path(__file__).resolve()),
               '--input-archive',str(a.input_archive),'--optics-archive',str(a.optics_archive),'--output',str(a.output)]))
        for i,entry in cases:
            if (a.output/'STOP').exists():queue['status']='stopped_by_user';save();return
            if any(b.sha256_file(REPO/p)!=h for p,h in identity['code'].items()):
                queue['status']='blocked_code_changed';save();raise ValueError('Code changed during batch')
            scene=entry['id'];run=a.output/scene;run.mkdir(exist_ok=True)
            sp=run/'batch_status.json'
            if sp.exists() and b.read_json(sp).get('status') in ('passed','failed_gate','failed'):continue
            status=dict(scene=scene,status='running');b.write_json(sp,status)
            queue.update(active=scene,active_stage='case pipeline: stage.json is live');save()
            try:
                b.invoke([sys.executable,str(HERE/'site_intensity_run.py'),'--run',str(run),
                          '--archive',str(a.optics_archive),'--truth',str(b.phase_path_for(entry,extracted)),
                          '--scene',scene,'--seed',str(674000+i)],run/'batch.log')
                ev=b.read_json(run/'evaluation.json')
                if ev['phase_sha256']!=b.array_sha(run/'final_phase.npy'):raise ValueError('Final phase hash mismatch')
                status.update(status='passed' if ev['quality_pass'] else 'failed_gate',evaluation=ev)
                dest=TRACK/'결과/site_reference_660_20261117'/scene;dest.mkdir(parents=True,exist_ok=True)
                for name in ['comparison.png','evaluation.json','manifest.json','gradient_validation.json']:
                    shutil.copy2(run/name,dest/name)
                idx=dest.parent/'results.json';summary=b.read_json(idx) if idx.exists() else {'reference_url':queue['reference_url'],'results':[]}
                summary['results']=[x for x in summary['results'] if x['scene']!=scene]+[ev];b.write_json(idx,summary)
            except Exception as exc:status.update(status='failed',error=str(exc))
            b.write_json(sp,status)
            queue['results']=[x for x in queue['results'] if x['scene']!=scene]+[status]
            save();print(json.dumps(status,ensure_ascii=False),flush=True)
        queue.update(active=None,status='completed' if all(x['status']=='passed' for x in queue['results']) else 'completed_with_failures');save()


if __name__=='__main__':main()
