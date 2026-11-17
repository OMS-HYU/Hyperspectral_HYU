"""Record preserved checkpoint after stopping only the audited batch process tree."""
import time
from pathlib import Path
import numpy as np
import batch_runner as b

root=Path(__file__).resolve().parent.parent/'실험/all_photos_660_float64_20261117'
with b.batch_lock(root/'batch.lock'):
    q=b.read_json(root/'queue.json')
    run=root/q['active']
    with np.load(run/'gd_state.npz',allow_pickle=False) as state:
        accepted=int(state['total'])
    progress=b.read_json(run/'progress.json')
    if accepted!=progress['gd_accepted']:raise ValueError('Derived GD progress differs from atomic state')
    checkpoint=dict(scene=q['active'],gd_accepted=accepted,state_file_sha256=b.sha256_file(run/'gd_state.npz'),
                    phase_sha256=b.array_sha(run/'latest_phase.npy'),
                    sensor_sha256=b.array_sha(run/'sensor.npy'))
    status=b.read_json(run/'batch_status.json')
    status.update(status='interrupted_for_audit',reason='User challenged speckle figure; observation model audit',checkpoint=checkpoint)
    b.write_json(run/'batch_status.json',status)
    q.update(status='paused_for_speckle_audit',checkpoint=checkpoint,
             updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    b.write_json(root/'queue.json',q)
    b.write_json(root/'speckle_audit_pause.json',dict(reason=status['reason'],checkpoint=checkpoint,
                 completed_results_preserved=True,model_and_solver_unchanged=True))
    print(checkpoint)
