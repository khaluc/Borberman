from __future__ import annotations
import json, threading, time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from agent import BomberlandAgent
from local_engine import LocalBomberland
from ppo_agent import PPOAgent
from strategies import AggressiveAgent, DefensiveAgent

HOST, PORT = "127.0.0.1", 8001
ROOT = Path(__file__).resolve().parent

class WebGame:
    def __init__(self):
        self.lock=threading.Lock(); self.engine=LocalBomberland(player_count=3)
        self.agents=self._agents(); self.running=False; self.finished_ticks=0
        threading.Thread(target=self._loop,daemon=True).start()
    def _loop(self):
        while True:
            time.sleep(.11)
            with self.lock:
                if not self.running: continue
                if self.engine.finished(400):
                    self.finished_ticks += 1
                    if self.finished_ticks >= 14:
                        self.engine=LocalBomberland(seed=int(time.time()),player_count=3)
                        self.agents=self._agents()
                        self.finished_ticks=0
                    continue
                state=self.engine.state()
                actions={key:self.agents[key].next_action(state,key) for key,p in self.engine.players.items() if p.alive}
                self.engine.step(actions)
    def _agents(self):
        model=ROOT/"bomberland_ppo.pt"
        main=PPOAgent(model,stochastic=True) if model.exists() else BomberlandAgent()
        return {
            "agent_1": main,
            "agent_2": DefensiveAgent(),
            "agent_3": AggressiveAgent(),
        }
    def _snapshot(self):
        state=self.engine.state(); state.update(running=self.running,finished=self.engine.finished(400)); return state
    def snapshot(self):
        with self.lock: return self._snapshot()
    def run(self,value):
        with self.lock: self.running=value; return self._snapshot()
    def reset(self):
        with self.lock:
            self.engine=LocalBomberland(seed=int(time.time()),player_count=3); self.agents=self._agents(); self.running=False; self.finished_ticks=0
            return self._snapshot()

game=WebGame()
class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs): super().__init__(*args,directory=str(ROOT),**kwargs)
    def do_GET(self):
        if urlparse(self.path).path=="/api/game": self.send_json(game.snapshot())
        else: super().do_GET()
    def do_POST(self):
        route=urlparse(self.path).path
        if route=="/api/game/reset": self.send_json(game.reset()); return
        if route=="/api/game/run":
            try:
                size=int(self.headers.get("Content-Length","0")); data=json.loads(self.rfile.read(size) or b"{}")
                self.send_json(game.run(bool(data.get("running"))))
            except (ValueError,json.JSONDecodeError): self.send_json({"error":"Invalid JSON"},HTTPStatus.BAD_REQUEST)
            return
        self.send_error(HTTPStatus.NOT_FOUND)
    def send_json(self,payload,status=HTTPStatus.OK):
        body=json.dumps(payload).encode(); self.send_response(status); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(body))); self.send_header("Cache-Control","no-store"); self.end_headers(); self.wfile.write(body)

if __name__=="__main__":
    print(f"Bomberland web: http://{HOST}:{PORT}")
    ThreadingHTTPServer((HOST,PORT),Handler).serve_forever()
