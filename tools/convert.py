import re, sys, nbformat
src = open(sys.argv[1]).read()
parts = re.split(r'^# %%(?: \[markdown\])?\s*$', src, flags=re.M)
heads = re.findall(r'^# %%( \[markdown\])?\s*$', src, flags=re.M)
cells=[]
for h, body in zip(heads, parts[1:]):
    body = body.strip('\n')
    if h: cells.append(nbformat.v4.new_markdown_cell('\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.split('\n'))))
    else: cells.append(nbformat.v4.new_code_cell(body))
nb = nbformat.v4.new_notebook(); nb.cells = cells
nb.metadata = {"accelerator":"GPU","colab":{"provenance":[],"gpuType":"T4"},"kaggle":{"accelerator":"nvidiaTeslaT4","isInternetEnabled":True},"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}}
nbformat.validate(nb); nbformat.write(nb, sys.argv[2]); print(len(cells),"cells")
