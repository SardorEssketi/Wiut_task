import time
import nbformat
from nbclient import NotebookClient
from .data import ROOT, ARTIFACTS, save_json


def execute():
    path=ROOT/'solution.ipynb'
    notebook=nbformat.read(path,as_version=4)
    start=time.time()
    def starting(cell,cell_index):
        if cell.cell_type=='code':
            print(f'Executing cell {cell_index+1}/{len(notebook.cells)}: {cell.source.splitlines()[0]}',flush=True)
    client=NotebookClient(notebook,timeout=2400,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}},on_cell_start=starting)
    try:
        client.execute()
    except Exception as exc:
        nbformat.write(notebook,path)
        save_json({'success':False,'seconds':time.time()-start,'error':str(exc)},ARTIFACTS/'notebook_execution.json')
        raise
    nbformat.write(notebook,path)
    assert all(output.get('output_type')!='error' for cell in notebook.cells if cell.cell_type=='code' for output in cell.get('outputs',[]))
    save_json({'success':True,'seconds':time.time()-start,'executed_code_cells':sum(c.cell_type=='code' for c in notebook.cells)},ARTIFACTS/'notebook_execution.json')
    print(f'Notebook executed successfully in {time.time()-start:.1f}s',flush=True)


if __name__=='__main__':
    execute()
