"""Ordinary-photo queue; RBC is explicitly routed to its separate method family."""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
import batch_runner as b

HERE=Path(__file__).resolve().parent;TRACK=HERE.parent
EXCLUDED={'mandrill','camera_rot90','barbara','boat','tank'}
FROZEN=['ordinary_policy_batch.py','ordinary_case_policy.py','evaluate_precision_case.py','render_five_panel.py',
        'native_cli.py','native_model.py','tank_precision_recovery.py','batch_runner.py']


def route(entries):
    ordinary=[];rbc=[]
    for i,item in enumerate(entries):
        if 'RBC' in item.get('name',''):
            microscopy=item['id'] in ['brightfield','darkfield','phasecontrast']
            rbc.append(dict(id=item['id'],name=item['name'],
                method_family='microscopy-image phase: separate weak-circular validation' if microscopy else 'Full GD100 + Spectral Periodic L-BFGS100',
                status='pending_native_method_validation',ordinary_pyramid_dispatch=False))
        elif item['id'] not in EXCLUDED:ordinary.append((i,item))
    return ordinary,rbc


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ['input-archive','optics-archive','output']:ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args()
    for k in ['input_archive','optics_archive','output']:setattr(a,k,getattr(a,k).resolve())
    a.output.mkdir(parents=True,exist_ok=True);os.environ['MPLCONFIGDIR']=str(a.output/'matplotlib_cache')
    with b.batch_lock(a.output/'batch.lock'):
        if b.sha256_file(a.optics_archive)!=b.EXPECTED_OPTICS:raise ValueError('Optics hash mismatch')
        extracted,_,entries=b.inspect_input(a.input_archive,a.output/'_source')
        cases,rbc=route(entries)
        b.write_json(a.output/'rbc_queue.json',dict(status='separate_method_validation_required',cases=rbc,
                      physics='same latest native5713 amplitude geometry; historical3800 results are not new validation',
                      evaluation='Physical RBC requires ROI/layer/group metrics; image-like microscopy inputs are separate'))
        identity=dict(code={n:b.sha256_file(HERE/n) for n in FROZEN},
                    input_sha256=b.sha256_file(a.input_archive),optics_sha256=b.EXPECTED_OPTICS,
                    gd_max=150,lbfgs_max=150,truth_used_for_stopping=False,cases=[x['id'] for i,x in cases])
        ip=a.output/'ordinary_policy_identity.json'
        if ip.exists() and b.read_json(ip)!=identity:raise ValueError('Ordinary policy identity changed')
        b.write_json(ip,identity)
        qp=a.output/'queue.json';old=b.read_json(qp) if qp.exists() else {}
        if old and not (a.output/'queue.before_ordinary_policy.json').exists():
            b.write_json(a.output/'queue.before_ordinary_policy.json',old)
        results=[x for x in old.get('results',[]) if x['scene'] in identity['cases']]
        queue=dict(status='running',pid=os.getpid(),total=len(cases),scope='ordinary photos',results=results,
                   cases=[dict(id=x['id'],seed=673000+i,name=x['name']) for i,x in cases],
                   rbc_separate_count=len(rbc),rbc_queue='rbc_queue.json',gd_max=150,lbfgs_max=150,
                   early_stop='sensor residual + objective stability + phase stability; no truth',
                   figure_order=['original','observed speckle','true phase','estimated phase','wrapped error'])
        def save():queue['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());b.write_json(qp,queue)
        save()
        b.write_json(a.output/'resume_command.json',dict(python=sys.executable,args=[str(Path(__file__).resolve()),
          '--input-archive',str(a.input_archive),'--optics-archive',str(a.optics_archive),'--output',str(a.output)]))
        for i,entry in cases:
            if (a.output/'STOP').exists():queue['status']='stopped_by_user';save();return
            if any(b.sha256_file(HERE/n)!=h for n,h in identity['code'].items()):
                queue['status']='blocked_code_changed';save();raise ValueError('Code changed during batch')
            scene=entry['id'];run=a.output/scene;sp=run/'batch_status.json'
            if sp.exists() and b.read_json(sp).get('status') in ['passed','failed_gate','failed']:continue
            run.mkdir(exist_ok=True);status=dict(scene=scene,status='running',stage='ordinary_150_150_adaptive')
            b.write_json(sp,status);queue.update(active=scene,active_stage='ordinary_150_150_adaptive');save()
            try:
                b.invoke([sys.executable,str(HERE/'ordinary_case_policy.py'),'--run',str(run),
                    '--archive',str(a.optics_archive),'--truth',str(b.phase_path_for(entry,extracted)),
                    '--scene',scene,'--seed',str(673000+i)],run/'ordinary_policy.log')
                ev=b.read_json(run/'recovery_float64/evaluation.json')
                checkpoint=run/'recovery_float64'/f"phase_{ev['lbfgs_accepted']:03}.npy"
                if ev['phase_sha256']!=b.array_sha(checkpoint):raise ValueError('Final checkpoint hash mismatch')
                status.update(status='passed' if ev['quality_pass'] else 'failed_gate',evaluation=ev)
                dest=TRACK/'결과/all_photos_660_float64_20261117'/scene;dest.mkdir(parents=True,exist_ok=True)
                for source in [run/'recovery_float64'/n for n in ['evaluation.json','comparison.png','config.json','precision_validation.json']]+[run/'completion_policy.json']:
                    shutil.copy2(source,dest/source.name)
                idx=dest.parent/'results.json';summary=b.read_json(idx) if idx.exists() else {'results':[]}
                summary['results']=[x for x in summary['results'] if x['scene']!=scene]+[ev];b.write_json(idx,summary)
            except Exception as exc:status.update(status='failed',error=str(exc))
            b.write_json(sp,status);queue['results']=[x for x in queue['results'] if x['scene']!=scene]+[status]
            save();print(json.dumps(status,ensure_ascii=False),flush=True)
        queue.update(active=None,status='ordinary_completed_rbc_separate');save()


if __name__=='__main__':main()
