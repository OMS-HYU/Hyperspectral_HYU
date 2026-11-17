"""Sequential 660 nm batch for all remaining manifest inputs, including RBC last.

User authorized all photos on 2026-11-17. Failures are recorded, then the queue
continues. RBC results are synthetic full-field phase tests, not measured QPI.
The established reference four and completed Tank precision variant are skipped.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import batch_runner as b

HERE=Path(__file__).resolve().parent
TRACK=HERE.parent
FROZEN=['native_cli.py','native_model.py','tank_precision_recovery.py','evaluate_precision_case.py','render_five_panel.py','all_photos_batch.py']


def plan(entries, summary):
    completed={x['id'] for x in summary['ordinary'] if x.get('known') and x['wavelength_nm']==660
               and x['gd_accepted']==150 and x['lbfgs_accepted']==60
               and x['final_circular_rmse_rad']<=.15 and x['sensor_l2']<=.025}
    completed.update(x['scene'] for x in summary.get('precision_recovery_experiments',[])
                     if x.get('quality_pass') and x.get('wavelength_nm')==660
                     and x.get('recovery_float64_lbfgs_accepted')==60)
    remaining=[x for x in entries if x['id'] not in completed]
    remaining.sort(key=lambda x:'RBC' in x.get('name',''))
    return remaining, sorted(completed)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-archive',type=Path,required=True)
    ap.add_argument('--optics-archive',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--inspect-only',action='store_true')
    a=ap.parse_args()
    for k in ['input_archive','optics_archive','output']:setattr(a,k,getattr(a,k).resolve())
    a.output.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(a.output/'matplotlib_cache')
    with b.batch_lock(a.output/'batch.lock'):
        if b.sha256_file(a.optics_archive)!=b.EXPECTED_OPTICS:raise ValueError('Optics archive mismatch')
        extracted,manifest,entries=b.inspect_input(a.input_archive,a.output/'_source')
        summary=b.read_json(TRACK/'결과/최신_20261116/native_results.json')
        cases,skipped=plan(entries,summary)
        identity=dict(input_sha256=b.sha256_file(a.input_archive),optics_sha256=b.EXPECTED_OPTICS,
                      code={name:b.sha256_file(HERE/name) for name in FROZEN},
                      cases=[x['id'] for x in cases],wavelength_nm=660,seed_base=673000)
        ip=a.output/'identity.json'
        if ip.exists() and b.read_json(ip)!=identity:raise ValueError('Batch identity changed; do not mix states')
        b.write_json(ip,identity)
        queue=dict(status='ready',pid=os.getpid(),total=len(cases),skipped_completed=skipped,
                   cases=[dict(id=x['id'],name=x.get('name'),source=x.get('source'),
                               seed=673000+entries.index(x),category='RBC synthetic phase' if 'RBC' in x.get('name','') else 'ordinary') for x in cases],results=[])
        qp=a.output/'queue.json'
        if qp.exists():queue['results']=b.read_json(qp).get('results',[])
        def save():
            queue['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
            b.write_json(qp,queue)
        save()
        print(json.dumps(queue,ensure_ascii=False),flush=True)
        if a.inspect_only:return 0
        b.write_json(a.output/'resume_command.json',dict(python=sys.executable,
                     args=[str(Path(__file__).resolve()),'--input-archive',str(a.input_archive),
                           '--optics-archive',str(a.optics_archive),'--output',str(a.output)]))
        for entry,meta in zip(cases,queue['cases']):
            if (a.output/'STOP').exists():queue['status']='stopped_by_user';save();return 0
            if any(b.sha256_file(HERE/n)!=h for n,h in identity['code'].items()):
                queue['status']='blocked_code_changed';save();raise RuntimeError('Code changed during batch')
            run=a.output/entry['id'];run.mkdir(exist_ok=True)
            statusfile=run/'batch_status.json'
            if statusfile.exists() and b.read_json(statusfile).get('status') in ('passed','failed_gate','failed'):
                continue
            status=dict(scene=entry['id'],status='running',stage='generate',seed=meta['seed'])
            queue.update(status='running',active=entry['id']);save()
            def stage(name):
                status['stage']=name;b.write_json(statusfile,status)
                queue['active_stage']=name;save()
            base=[sys.executable,str(TRACK/'run.py')]
            common=['--run',str(run),'--archive',str(a.optics_archive)]
            log=run/'batch.log'
            try:
                truth=b.phase_path_for(entry,extracted)
                if (run/'manifest.json').exists():
                    man=b.read_json(run/'manifest.json')
                    expected=dict(scene=entry['id'],noise_seed=meta['seed'],wavelength_nm=660,
                                  truth_sha256=b.array_sha(truth),archive_sha256=b.EXPECTED_OPTICS)
                    if any(man.get(k)!=v for k,v in expected.items()):raise ValueError('Run input identity mismatch')
                else:
                    if any((run/n).exists() for n in ['sensor.npy','private_truth.npy']):
                        raise RuntimeError('Partial generation preserved; manual recovery required for this case')
                    stage('generate');b.invoke(base+['generate',*common,'--truth',str(truth),
                                               '--scene',entry['id'],'--seed',str(meta['seed']),'--wavelength','660'],log)
                stage('gd150')
                done=0
                if (run/'gd_state.npz').exists():
                    with np.load(run/'gd_state.npz',allow_pickle=False) as z:done=int(z['total'])
                if done<150:b.invoke(base+['gd',*common,'--steps',str(150-done)],log)
                if b.read_json(run/'progress.json')['gd_accepted']!=150:raise RuntimeError('GD150 incomplete')
                stage('float64_lbfgs60')
                b.invoke([sys.executable,str(HERE/'tank_precision_recovery.py'),*common,'--until','60'],log)
                if b.read_json(run/'recovery_float64/progress.json')['accepted']!=60:raise RuntimeError('L60 incomplete')
                stage('evaluate')
                if not (run/'recovery_float64/evaluation.json').exists():
                    b.invoke([sys.executable,str(HERE/'evaluate_precision_case.py'),*common],log)
                ev=b.read_json(run/'recovery_float64/evaluation.json')
                if ev['phase_sha256']!=b.array_sha(run/'recovery_float64/phase_060.npy'):
                    raise ValueError('Final phase hash mismatch')
                status.update(status='passed' if ev['quality_pass'] else 'failed_gate',evaluation=ev)
                dest=TRACK/'결과/all_photos_660_float64_20261117'/entry['id']
                dest.mkdir(parents=True,exist_ok=True)
                for name in ['evaluation.json','comparison.png','config.json','precision_validation.json']:
                    shutil.copy2(run/'recovery_float64'/name,dest/name)
                # Variant results are deliberately separate from the common float32 leaderboard.
                result_index=dest.parent/'results.json'
                index=b.read_json(result_index) if result_index.exists() else {'variant':'float64 L60','results':[]}
                index['results']=[x for x in index['results'] if x['scene']!=entry['id']]+[ev]
                b.write_json(result_index,index)
            except Exception as exc:
                status.update(status='failed',error=str(exc))
            status['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
            b.write_json(statusfile,status)
            queue['results']=[x for x in queue['results'] if x['scene']!=entry['id']]+[status]
            save();print(json.dumps(status,ensure_ascii=False),flush=True)
        queue.update(status='completed' if all(x['status']=='passed' for x in queue['results']) else 'completed_with_failures',active=None)
        save()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
