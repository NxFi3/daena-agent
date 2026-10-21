"""Exercise the installed wheel outside the source checkout, without inference."""
import json
from pathlib import Path
import time
from tempfile import TemporaryDirectory
import sys

from PySide6.QtWidgets import QApplication
import src
from src.agent.agent import Agent
from src.models.LLMResult import LLMResult
from src.studio.window import StudioWindow

checkout=Path(sys.argv[1]).resolve()
assert not Path(src.__file__).resolve().is_relative_to(checkout), src.__file__
assert src.__version__=="0.6.0"


class Model:
    model=None
    def __init__(self):self.calls=0
    def generate(self,messages,**kwargs):
        self.calls+=1
        if self.calls<3:
            name,args=("apply_patch",{"patch":"*** Begin Patch\n*** Add File: wheel.txt\n+installed desktop works\n*** End Patch"}) if self.calls==1 else ("read_file",{"file_path":"wheel.txt"})
            return LLMResult("",{},[{"id":str(self.calls),"type":"function","function":{"name":name,"arguments":json.dumps(args)}}],None,1)
        return LLMResult("Created and checked wheel.txt",{},[],None,1)


app=QApplication([])
with TemporaryDirectory() as directory:
    window=StudioWindow(directory)
    model=Model();window.agent_factory=lambda c:Agent(c,llm=model)
    window.input.setPlainText("Create wheel.txt and read it back")
    window.send()
    deadline=time.monotonic()+10
    while window.busy and time.monotonic()<deadline:
        app.processEvents();time.sleep(.01)
    assert not window.busy, "Desktop worker did not finish"
    assert window._result["metrics"]["completed"],window._result
    assert (Path(directory)/"wheel.txt").read_text().strip()=="installed desktop works"
    assert "Created and checked wheel.txt" in window.transcript.toPlainText()
    assert window.history.count()==1
    assert window.execution.provider.currentText()=="ollama"
    window.close();app.processEvents()
print("INSTALLED_STUDIO_WHEEL_OK")
