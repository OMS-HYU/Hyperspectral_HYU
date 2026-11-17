"""Figure-only revision after reading the user's authoritative single chat."""
import os
import shutil
from pathlib import Path
import batch_runner as b
import all_photos_batch as batch
from render_five_panel import render

root=batch.TRACK/'실험/all_photos_660_float64_20261117'
os.environ['MPLCONFIGDIR']=str(root/'matplotlib_cache')
with b.batch_lock(root/'batch.lock'):
    q=b.read_json(root/'queue.json')
    if q['status']!='paused_for_speckle_audit' or not (root/'STOP').exists():
        raise RuntimeError('Requires preserved native batch audit pause')
    identity=b.read_json(root/'identity.json')
    for name in ['native_cli.py','native_model.py','tank_precision_recovery.py']:
        if b.sha256_file(batch.HERE/name)!=identity['code'][name]:
            raise ValueError('Physics/solver code differs from preserved native run: '+name)
    current=root/q['active']
    if b.sha256_file(current/'gd_state.npz')!=q['checkpoint']['state_file_sha256']:
        raise ValueError('Atomic checkpoint changed since pause')
    backup=root/'identity.before_latest_chat_revision.json'
    if backup.exists():raise FileExistsError(backup)
    shutil.copy2(root/'identity.json',backup)
    resultroot=batch.TRACK/'결과/all_photos_660_float64_20261117'
    index=b.read_json(resultroot/'results.json')
    for result in q['results']:
        scene=result['scene'];run=root/scene;out=run/'recovery_float64'
        ev=b.read_json(out/'evaluation.json')
        phase_before=b.array_sha(out/'phase_060.npy');sensor_before=b.array_sha(run/'sensor.npy')
        old=out/'comparison.before_latest_chat_style.png'
        if not old.exists():shutil.copy2(out/'comparison.png',old)
        ev['figure']=render(run,out/'comparison.png')
        if b.array_sha(out/'phase_060.npy')!=phase_before or b.array_sha(run/'sensor.npy')!=sensor_before:
            raise RuntimeError('Rendering altered physical arrays')
        b.write_json(out/'evaluation.json',ev)
        for name in ['comparison.png','evaluation.json','comparison.before_latest_chat_style.png']:
            shutil.copy2(out/name,resultroot/scene/name)
        result['evaluation']=ev
        status=b.read_json(run/'batch_status.json');status['evaluation']=ev;b.write_json(run/'batch_status.json',status)
        index['results']=[x for x in index['results'] if x['scene']!=scene]+[ev]
    b.write_json(resultroot/'results.json',index)
    identity['code']={name:b.sha256_file(batch.HERE/name) for name in batch.FROZEN}
    b.write_json(root/'identity.json',identity)
    status=b.read_json(current/'batch_status.json');status.update(status='ready_to_resume',reason='Latest single chat confirms original native physics')
    b.write_json(current/'batch_status.json',status)
    q.update(status='ready_to_resume',physics_authority='ChatGPT 싱글스펙 6afbf72d-51c4-83ee-85c2-82c0c08e46e2',
             figure_layout='original / observed amplitude inferno / true phase viridis / estimate viridis / wrapped error')
    b.write_json(root/'queue.json',q)
    b.write_json(root/'latest_chat_revision.json',dict(thread_id='6afbf72d-51c4-83ee-85c2-82c0c08e46e2',
                 change='figure style only; original native model/solver byte hashes verified',
                 preserved_checkpoint=q['checkpoint'],physical_arrays_unchanged=True))
    (root/'STOP').rename(root/'STOP.latest_chat_revision_applied')
    print('Native model/solver unchanged; GD53 preserved; figure-only revision ready to resume',flush=True)
