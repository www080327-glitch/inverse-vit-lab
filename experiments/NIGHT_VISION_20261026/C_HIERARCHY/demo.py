"""Inference-only executable. Accepts image + task ID; cannot request oracle."""
import argparse
from pathlib import Path
from PIL import Image
from runtime import ROOT, torch, initialize, write_json
from memory import PixelMemory
from model import Hierarchy

def load_model():
    model=Hierarchy()
    model.load_state_dict(torch.load(ROOT/'results/model.pt',map_location='cpu',weights_only=True))
    model.eval()
    return model

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--image',default='data/images/test/test_0000.png')
    parser.add_argument('--question',type=int,choices=range(4),default=0)
    parser.add_argument('--k',type=int,default=2)
    parser.add_argument('--method',choices=['learned','uniform','random'],default='learned')
    args=parser.parse_args(); initialize()
    image_path=Path(args.image)
    if not image_path.is_absolute(): image_path=ROOT/image_path
    memory=PixelMemory.pack(Image.open(image_path).convert('RGB')); model=load_model()
    with torch.inference_mode(): out=model(memory,args.question,args.k,args.method)
    public_image=str(image_path.relative_to(ROOT)) if image_path.is_relative_to(ROOT) else image_path.name
    result=dict(image=public_image,question_id=args.question,selected_regions=out['ids'].tolist(),
                selected_pixel_positions=out['selected_fine_positions'],
                raw_pixel_count=memory.height*memory.width,storage=memory.storage(),
                answer_id=int(out['logits'].argmax()),answer_probability=out['logits'].softmax(-1).tolist(),
                coarse_scores=out['coarse']['scores'].tolist(),coarse_context=out['coarse']['context'].tolist(),
                attention_sum=float(out['weights'].sum()),output=out['output'].tolist(),
                note='Canonical task ID only. No natural-language model. Labels were not loaded.')
    write_json(ROOT/'results/demo.json',result)
    import json
    print(json.dumps({k:v for k,v in result.items() if k not in ['coarse_context','output']},indent=2))

if __name__=='__main__': main()
