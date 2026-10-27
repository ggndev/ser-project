import sys, re, os
src = open(sys.argv[1]).read(); root = sys.argv[2]
src = src.replace("QUICK_RUN   = False","QUICK_RUN   = True").replace('"facebook/wav2vec2-base"','"tiny_w2v"').replace('"facebook/hubert-base-ls960"','"tiny_hubert"').replace('"microsoft/wavlm-base-plus"','"tiny_wavlm"')
code = ["\n".join(l for l in c.split("\n") if not l.startswith(("!","%"))) for c in re.split(r'^# %%.*$', src, flags=re.M)[1:]]
import matplotlib; matplotlib.use("Agg"); os.chdir(root)
from IPython.display import display
g = {"__name__":"__main__","display":display}
for i,c in enumerate(code):
    try: exec(compile(c, f"cell{i}", "exec"), g)
    except Exception:
        import traceback; traceback.print_exc(); print("FAILED in cell", i); sys.exit(1)
print("ALL CELLS OK")
