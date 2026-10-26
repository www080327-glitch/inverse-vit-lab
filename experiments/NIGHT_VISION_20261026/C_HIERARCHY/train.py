"""Train tiny models from TRAIN annotations only; persistent cumulative budget."""
import time
import json
import os
from PIL import Image
from runtime import ROOT, torch, np, initialize, write_json, read_jsonl
from memory import PixelMemory
from model import Hierarchy
from torch.nn import functional as F

def train():
    initialize()
    if (ROOT/'results/model.pt').exists():
        print('Existing model kept. No additional training.'); return
    lock = ROOT/'results/training.lock'
    fd = os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.close(fd)
    budget_path = ROOT/'results/training_budget.json'
    budget = json.loads(budget_path.read_text()) if budget_path.exists() else dict(limit_seconds=1200,runs=[])
    used = sum(r['charged_seconds'] for r in budget['runs'])
    reservation = min(180,1200-used)
    if reservation<=0:
        lock.unlink(); raise RuntimeError('Cumulative synthetic training budget exhausted')
    run = dict(start_unix=time.time(),charged_seconds=reservation,status='reserved_including_crash')
    budget['runs'].append(run); write_json(budget_path,budget)
    start=time.perf_counter(); stages=[]
    def check():
        if time.perf_counter()-start>reservation-1: raise TimeoutError('Training time reservation reached')
    try:
        scenes=[]; annotations=[]
        for split in ['train','train_blind']:
            rows=read_jsonl(ROOT/f'data/inputs/{split}.jsonl')
            anns=read_jsonl(ROOT/f'data/annotations/{split}.jsonl')
            for row,ann in zip(rows,anns):
                assert row['id']==ann['id']
                scenes.append(PixelMemory.pack(Image.open(ROOT/row['image'])))
                annotations.append(ann)
        tiles=torch.stack([s.pixels for s in scenes]); summaries=torch.stack([s.summary() for s in scenes])
        targets=torch.tensor([a['regions'] for a in annotations])
        kinds=torch.tensor([a['region_kinds'] for a in annotations])
        model=Hierarchy(); gen=torch.Generator().manual_seed(361)
        def optimize(name,params,steps,lossfn,lr=.008):
            t=time.perf_counter(); opt=torch.optim.Adam(params,lr=lr); first=None
            for step in range(steps):
                check(); opt.zero_grad(); loss=lossfn(); loss.backward(); opt.step()
                if first is None: first=float(loss.detach())
            stages.append(dict(name=name,steps=steps,first_loss=first,last_loss=float(loss.detach()),seconds=time.perf_counter()-t))
        def route_loss():
            ids=torch.randint(len(scenes),(48,),generator=gen)
            scores=model.coarse(summaries[ids]).transpose(1,2).reshape(-1,16)
            return F.cross_entropy(scores,targets[ids].reshape(-1))
        optimize('coarse_router_train_region_supervision',model.coarse.parameters(),700,route_loss)
        x=model.normalize(tiles).reshape(-1,64); labels=kinds.reshape(-1)
        # Uniform type sampling avoids the 12/16 distractor majority shortcut.
        bytype=[torch.where(labels==i)[0] for i in range(5)]
        def fine_loss():
            ids=torch.cat([group[torch.randint(len(group),(32,),generator=gen)] for group in bytype])
            return F.cross_entropy(model.fine_type(x[ids]),labels[ids])
        optimize('fine_region_type_only_no_answer_labels',model.fine_type.parameters(),900,fine_loss)
        # Answer head sees actual attention outputs, not labels as inputs.
        # Train with random supersets that include the supervised TRAIN target.
        t=time.perf_counter(); outputs=[]; answers=[]
        with torch.inference_mode():
            for scene,ann in zip(scenes,annotations):
                for q in range(4):
                    check(); count=int(torch.tensor([1,2,4,16])[torch.randint(4,(),generator=gen)])
                    ids=torch.randperm(16,generator=gen)[:count].tolist()
                    ids[0]=ann['regions'][q]
                    sel,valid,_=scene.gather(ids)
                    out=model.read_tiles(sel,q,valid)['output']
                    outputs.append(torch.cat((out,F.one_hot(torch.tensor(q),4).float())))
                    answers.append(ann['answers'][q])
        # clone outside inference mode so autograd may save inputs.
        features=torch.stack(outputs).clone(); answer_labels=torch.tensor(answers)
        stages.append(dict(name='prepare_train_attention_outputs',seconds=time.perf_counter()-t,examples=len(features)))
        def answer_loss():
            ids=torch.randint(len(features),(128,),generator=gen)
            return F.cross_entropy(model.answer(features[ids]),answer_labels[ids])
        optimize('answer_head_train_answers_only',model.answer.parameters(),1000,answer_loss,.01)
        model.eval()
        with torch.inference_mode():
            fine_acc=(model.fine_type(x).argmax(-1)==labels).float().mean().item()
            answer_acc=(model.answer(features).argmax(-1)==answer_labels).float().mean().item()
        torch.save(model.state_dict(),ROOT/'results/model.pt')
        write_json(ROOT/'results/training.json',dict(environment=initialize(),stages=stages,
                   parameters=sum(p.numel() for p in model.parameters()),train_scenes=len(scenes),
                   fine_type_train_accuracy=fine_acc,answer_head_train_accuracy=answer_acc,
                   test_files_opened=False,training_supervision='TRAIN region IDs/type IDs; TRAIN answers for head only',
                   handcrafted_query='one-hot glyph task IDs, not natural-language understanding'))
        run['status']='completed'
    except BaseException:
        run['status']='failed'; raise
    finally:
        run['charged_seconds']=time.perf_counter()-start
        run['end_unix']=time.time(); budget['total_charged_seconds']=sum(r['charged_seconds'] for r in budget['runs'])
        write_json(budget_path,budget); lock.unlink()
    print(json.dumps(dict(stages=stages,total_training_seconds=budget['total_charged_seconds']),indent=2))

if __name__=='__main__': train()
