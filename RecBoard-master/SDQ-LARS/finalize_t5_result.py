import json, pickle, sys
from pathlib import Path
p=Path(sys.argv[1])
with (p/"data/best.pkl").open("rb") as f:
    scores=pickle.load(f)
result={"selection":"validation_NDCG@10","validation_best":scores.get("valid"),"test":scores.get("best"),"source":str(p/"data/best.pkl")}
(p/"result.json").write_text(json.dumps(result, indent=2))
# Mirror the selected-checkpoint result into the run directory for monitoring.
# Example log root: logs/LARS-20260926_cp_ug_hlars-T5/Amazon2014Sports_550_LOU/t5
name=p.parents[1].name
if name.startswith("LARS-") and name.endswith("-T5"):
    run_name=name[len("LARS-"):-len("-T5")]
    dataset=p.parent.name.removeprefix("Amazon2014").removesuffix("_550_LOU")
    out=Path(__file__).resolve().parent/"runs"/run_name/dataset/"result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
