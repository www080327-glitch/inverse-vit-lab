"""Create audited C13 archive; no datasets, weights, logs or recursive directory scan."""
import hashlib,json,zipfile
from pathlib import Path
p=Path(__file__).resolve().parent
names=["PLAN_ZH.md","MATH_ZH.md","HANDOFF_ZH.md","RESULT.json","PARAMETER_MAPPING.json",
       "experiment.py","math_examples.py","make_report.py","package_delivery.py",
       "requirements.txt","learning_curves.svg","reconstructions.svg","README.md"]
files={name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in names}
(p/"MANIFEST.sha256").write_text("".join(f"{v}  {k}\n" for k,v in files.items()))
with zipfile.ZipFile(p/"C13_DELIVERY.zip","w",compression=zipfile.ZIP_DEFLATED) as z:
    for name in names+["MANIFEST.sha256"]:
        z.write(p/name,arcname="C13/"+name)
with zipfile.ZipFile(p/"C13_DELIVERY.zip") as z:
    assert z.testzip() is None
print(json.dumps({"archive":"C13_DELIVERY.zip","bytes":(p/"C13_DELIVERY.zip").stat().st_size,
 "sha256":hashlib.sha256((p/"C13_DELIVERY.zip").read_bytes()).hexdigest(),"files":len(names)+1}))
