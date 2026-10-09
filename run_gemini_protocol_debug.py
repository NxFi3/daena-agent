from __future__ import annotations
import json, os, traceback
from pathlib import Path
from uuid import uuid4
ROOT=Path("/home/itsnxfi3/Desktop/daena-agent")
os.environ.setdefault("DAENA_CLI","1")
from dotenv import load_dotenv
load_dotenv(ROOT/".env",override=False)
from src.agent.agent import Agent
from src.agent.agentloop import Loop
from src.engine.providers.builtin.gemini.geminiprovider import GeminiProvider
from src.models.ContextEvent import ContextEvent,ContextPriority,ContextRole,ContextType
from cli.render import StreamRenderer

orig_generate=GeminiProvider.generate
def instrumented_generate(self,llminput):
    raw_rows=[]
    for i,m in enumerate(llminput.messages or []):
        calls=[]
        for call in m.get('tool_calls') or []:
            fn=call.get('function') if isinstance(call,dict) else {}
            fn=fn if isinstance(fn,dict) else {}
            calls.append({'id':call.get('id'),'name':fn.get('name'),'signature_present':bool(call.get('thought_signature_b64'))})
        raw_rows.append({'i':i,'role':m.get('role'),'name':m.get('name'),'tool_call_id':m.get('tool_call_id'),'tool_name':m.get('tool_name'),'has_content':bool(m.get('content')),'calls':calls})
    print('GEMINI_RAW_MESSAGES='+json.dumps(raw_rows,ensure_ascii=False),flush=True)
    converted, _ = self._convert_messages(llminput.messages or [])
    rows=[]
    for i,c in enumerate(converted):
        p=[]
        for part in c.parts or []:
            fc=getattr(part,"function_call",None)
            fr=getattr(part,"function_response",None)
            txt=getattr(part,"text",None)
            if fc is not None:
                p.append({"kind":"function_call","name":fc.name,"id":fc.id,"signature_present":bool(getattr(part,"thought_signature",None))})
            elif fr is not None:
                p.append({"kind":"function_response","name":fr.name,"id":fr.id})
            elif txt is not None:
                p.append({"kind":"text","prefix":str(txt)[:45]})
            else:
                p.append({"kind":"other"})
        rows.append({"i":i,"role":c.role,"parts":p})
    print("GEMINI_CONTENT_SEQUENCE="+json.dumps(rows,ensure_ascii=False),flush=True)
    try:
        return orig_generate(self,llminput)
    except Exception as exc:
        print("GEMINI_GENERATE_ERROR="+repr(exc),flush=True)
        raise
GeminiProvider.generate=instrumented_generate

config=json.loads((ROOT/"config.json").read_text(encoding="utf-8"))
config["llm"]={"provider":"gemini","provider_config":{"model_name":"gemini-3.8-flash","generation_config":{"temperature":0.2,"think":"low"}}}
config["max_agent_iterations"]=4
config["max_auto_continuations"]=0
config["completion_review"]={"enabled":False}
config["memory"]={"stm_db_path":str(ROOT/"data"/"stm.db")}
workspace=ROOT/"workspaces"/"house-price-dataset"
agent=Agent(config); agent.set_workingdirectory(str(workspace))
task=ContextEvent(role=ContextRole.USER,type=ContextType.MESSAGE,content="In the current workspace, call list_dir with path='.' exactly once. Then answer with the number of entries returned. Do not call any other tool.",priority=ContextPriority.NORMAL,step=0,metadata={"source":"gemini-protocol-diagnostic"})
renderer=StreamRenderer(model="gemini-3.8-flash",workspace=str(workspace),session_id=str(agent.session_id),think_enabled="low")
err=None; result=None
try:
    result=agent.act(task,on_event=renderer.handle)
except BaseException as exc:
    err=exc; traceback.print_exc()
finally:
    renderer.finish(result=result,error=err)
    print("DEBUG_METRICS="+json.dumps(agent.last_run_metrics,ensure_ascii=False,default=str),flush=True)
    agent.close()
