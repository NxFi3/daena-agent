from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

os.environ.setdefault("DAENA_CLI","1")

from src.agent.agent import Agent
from src.models.ContextEvent import ContextEvent, ContextPriority, ContextRole, ContextType
from .render import StreamRenderer

ROOT=Path(__file__).resolve().parent.parent
CONFIG_PATH=ROOT/"config.json"
STYLE=Style.from_dict({"prompt":"#A78BFA bold","completion-menu.completion":"bg:#1C1033 fg:#CBD5E1","completion-menu.completion.current":"bg:#4C1D95 fg:#F5F3FF bold"})
COMMANDS={"/help":None,"/status":None,"/new":None,"/reasoning":{"low":None,"medium":None,"high":None,"off":None,"auto":None},"/think":{"low":None,"medium":None,"high":None,"off":None,"auto":None},"/interrupt":None,"/clear":None,"/exit":None,"/busy":{"steer":None,"interrupt":None,"queue":None}}
BUSY_MODES={"steer","interrupt","queue"}

def load_config():
    with CONFIG_PATH.open("r",encoding="utf-8") as f:return json.load(f)

def think_value(agent):
    return (getattr(agent.llm,"generation_config",{}) or {}).get("think","medium")

def set_think(agent,value):
    agent.llm.generation_config["think"]=value

def command(agent,text,busy_mode,running):
    parts=text.split(maxsplit=1); cmd=parts[0].lower(); arg=parts[1].strip().lower() if len(parts)>1 else ""
    if cmd in {"/exit","/quit","/q"}: return "exit",busy_mode
    if cmd=="/help":
        print("  /status  /new  /reasoning low|medium|high|off|auto  /busy steer|interrupt|queue")
        print("  /interrupt  /clear  /exit")
    elif cmd=="/status":
        print(f"Daena  model={agent.llm.llm_config.get('model_name') or agent.llm.model.defaultModel} provider={agent.llm.provider_name} think={think_value(agent)} busy={busy_mode}")
        print(f"workspace={agent.workingdirectory}")
    elif cmd=="/new":
        from uuid import uuid4
        agent.session_id=uuid4(); agent.loop.session_id=agent.session_id
        print("new conversation started")
    elif cmd in {"/reasoning","/think"}:
        value=arg or str(think_value(agent)).lower()
        value=None if value=="auto" else False if value=="off" else value
        if value not in {None,False,"low","medium","high"}: print("usage: /reasoning low|medium|high|off|auto")
        else: set_think(agent,value); print(f"reasoning = {think_value(agent)}")
    elif cmd=="/busy":
        if arg in BUSY_MODES: busy_mode=arg; print(f"busy mode = {busy_mode}")
        else: print(f"busy mode = {busy_mode}")
    elif cmd=="/interrupt":
        stop=getattr(agent,"_cli_state",{}).get("stop_event")
        if stop: stop.set()
    elif cmd=="/clear":
        os.system("cls" if os.name=="nt" else "clear")
    else: print(f"unknown command: {cmd}")
    return "continue",busy_mode

def start_turn(agent,text,state):
    stop=threading.Event()
    renderer=StreamRenderer(str(agent.llm.llm_config.get("model_name") or agent.llm.model.defaultModel),agent.workingdirectory,str(agent.session_id),think_value(agent))
    state.update(running=True,stop_event=stop,renderer=renderer,result=None,error=None)

    task=ContextEvent(role=ContextRole.USER,type=ContextType.MESSAGE,content=text,priority=ContextPriority.NORMAL,step=0,metadata={"source":"cli","workspace":agent.workingdirectory})

    def emit(event):
        renderer.handle(event); state["toolbar"]=renderer.toolbar()

    def worker():
        try:
            state["result"]=agent.act(task,on_event=emit,stop_event=stop)
            renderer.handle({"type":"final_response","text":getattr(state["result"],"response","")})
        except Exception as exc:
            state["error"]=exc; renderer.handle({"type":"error","message":f"{type(exc).__name__}: {exc}"})
        finally:
            renderer.finish(state.get("result"),state.get("error"))
            state["running"]=False; state["stop_event"]=None; state["turn_finished"]=True

    state["turn_finished"]=False
    threading.Thread(target=worker,name="daena-agent",daemon=True).start()

def main():
    parser=argparse.ArgumentParser(prog="daena")
    parser.add_argument("--cwd",default=None); parser.add_argument("--no-think",action="store_true")
    args=parser.parse_args()
    agent=Agent(load_config())
    if args.cwd: agent.set_workingdirectory(args.cwd)
    if args.no_think: set_think(agent,False)
    print(f"\n  ◈ D A E N A  {agent.llm.llm_config.get('model_name') or agent.llm.model.defaultModel} · {agent.llm.provider_name}")
    print(f"  workspace: {agent.workingdirectory}\n")
    session=PromptSession(style=STYLE)
    completer=NestedCompleter.from_nested_dict(COMMANDS)
    state={"running":False,"stop_event":None,"renderer":None,"turn_finished":False,"toolbar":""}
    busy_mode="steer"
    try:
        while True:
            running=bool(state["running"])
            try:
                with patch_stdout(raw=True):
                    text=session.prompt(HTML(f"<ansibrightmagenta><b>you</b></ansibrightmagenta> <ansicyan>{'»' if running else '›'}</ansicyan> "),completer=completer,bottom_toolbar=lambda:state.get("toolbar") or "").strip()
            except KeyboardInterrupt:
                if state.get("stop_event"): state["stop_event"].set()
                continue
            except EOFError: break
            if not text: continue
            if text.startswith("/"):
                action,busy_mode=command(agent,text,busy_mode,running)
                if action=="exit": break
                continue
            if running:
                if busy_mode=="interrupt":
                    if state.get("stop_event"): state["stop_event"].set()
                    state["pending"]=text
                elif busy_mode=="queue":
                    state.setdefault("queue",[]).append(text)
                else:
                    agent.steer(text)
                continue
            start_turn(agent,text,state)
            while state.get("running") is True:
                import time; time.sleep(0.05)
            if state.pop("pending",None): start_turn(agent,state.pop("pending"),state)
            elif state.get("queue"): start_turn(agent,state["queue"].pop(0),state)
    finally:
        if state.get("stop_event"): state["stop_event"].set()
        agent.close()

if __name__=="__main__":
    main()
