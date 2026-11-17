"""Audited figure-only batch migration at a completed-case boundary."""
import argparse
import os
import shutil
from pathlib import Path
import all_photos_batch as batch
import batch_runner as b
from render_five_panel import render


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();root=a.output.resolve()
    os.environ['MPLCONFIGDIR']=str(root/'matplotlib_cache')
    with b.batch_lock(root/'batch.lock'):
        queue=b.read_json(root/'queue.json')
        if queue['status']!='stopped_by_user' or not (root/'STOP').exists():
            raise RuntimeError('Requires a graceful stop between cases')
        identity=b.read_json(root/'identity.json')
        for name in ['native_cli.py','native_model.py','tank_precision_recovery.py']:
            if b.sha256_file(batch.HERE/name)!=identity['code'][name]:
                raise ValueError('Optimizer or optical model changed; figure-only migration refused')
        backup=root/'identity.before_five_panel.json'
        if backup.exists():raise FileExistsError(backup)
        shutil.copy2(root/'identity.json',backup)
        hashes={name:b.sha256_file(batch.HERE/name) for name in batch.FROZEN}
        destroot=batch.TRACK/'결과/all_photos_660_float64_20261117'
        index=b.read_json(destroot/'results.json')
        for result in queue['results']:
            scene=result['scene'];run=root/scene;out=run/'recovery_float64'
            if not (out/'evaluation.json').exists():continue
            phase_hash=b.array_sha(out/'phase_060.npy')
            previous=out/'comparison_three_panel.png'
            if not previous.exists():shutil.copy2(out/'comparison.png',previous)
            figure=render(run,out/'comparison.png')
            ev=b.read_json(out/'evaluation.json');ev['figure']=figure
            if b.array_sha(out/'phase_060.npy')!=phase_hash or phase_hash!=ev['phase_sha256']:
                raise ValueError('Phase changed during rendering')
            b.write_json(out/'evaluation.json',ev)
            dest=destroot/scene
            for name in ['comparison.png','comparison_three_panel.png','evaluation.json']:
                shutil.copy2(out/name,dest/name)
            result['evaluation']=ev
            status=b.read_json(run/'batch_status.json');status['evaluation']=ev
            b.write_json(run/'batch_status.json',status)
            index['results']=[x for x in index['results'] if x['scene']!=scene]+[ev]
        b.write_json(destroot/'results.json',index)
        b.write_json(root/'figure_migration.json',dict(reason='User requested five panels; plotting only',
                     old_code=identity['code'],new_code=hashes,optimizer_unchanged=True,
                     original_comparisons_preserved=True,phase_hashes_verified=True))
        identity['code']=hashes;b.write_json(root/'identity.json',identity)
        queue['figure_layout']='original, speckle intensity, true phase, estimated phase, circular phase error'
        b.write_json(root/'queue.json',queue)
        (root/'STOP').rename(root/'STOP.figure_migration_applied')
        print('Five-panel migration verified; optimizer unchanged; batch ready to resume',flush=True)


if __name__=='__main__':main()
