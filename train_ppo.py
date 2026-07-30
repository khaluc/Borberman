"""Train one shared PPO policy through four-agent self-play."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from local_engine import LocalBomberland
from ppo_agent import ACTIONS, DEVICE, ActorCritic, encode_observation


def calculate_gae(rewards, values, dones, gamma=.99, gae_lambda=.95):
    advantages=np.zeros(len(rewards),dtype=np.float32); carry=0.0; next_value=0.0
    for index in reversed(range(len(rewards))):
        mask=1.0-float(dones[index])
        delta=rewards[index]+gamma*next_value*mask-values[index]
        carry=delta+gamma*gae_lambda*mask*carry
        advantages[index]=carry; next_value=values[index]
    return advantages, advantages+np.asarray(values,dtype=np.float32)


def update(model, optimizer, memory, epochs=4, batch_size=256):
    all_obs=[]; all_actions=[]; all_old=[]; all_adv=[]; all_returns=[]
    for trajectory in memory.values():
        if not trajectory["rewards"]: continue
        adv,returns=calculate_gae(trajectory["rewards"],trajectory["values"],trajectory["dones"])
        all_obs.extend(trajectory["obs"]); all_actions.extend(trajectory["actions"])
        all_old.extend(trajectory["logp"]); all_adv.extend(adv); all_returns.extend(returns)
    obs=torch.as_tensor(np.asarray(all_obs),dtype=torch.float32,device=DEVICE)
    actions=torch.as_tensor(all_actions,dtype=torch.long,device=DEVICE)
    old=torch.as_tensor(all_old,dtype=torch.float32,device=DEVICE)
    adv=torch.as_tensor(np.asarray(all_adv),dtype=torch.float32,device=DEVICE)
    returns=torch.as_tensor(np.asarray(all_returns),dtype=torch.float32,device=DEVICE)
    adv=(adv-adv.mean())/(adv.std()+1e-8); loss_value=0.0
    for _ in range(epochs):
        for batch in torch.randperm(len(actions),device=DEVICE).split(batch_size):
            distribution,values=model(obs[batch]); logp=distribution.log_prob(actions[batch])
            ratio=(logp-old[batch]).exp()
            policy=-torch.min(ratio*adv[batch],torch.clamp(ratio,.8,1.2)*adv[batch]).mean()
            value=nn.functional.mse_loss(values,returns[batch]); entropy=distribution.entropy().mean()
            loss=policy+.5*value-.02*entropy
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),.5); optimizer.step()
            loss_value=loss.item()
    return loss_value


def train(matches: int, output: Path, update_every: int = 8, max_steps: int = 300):
    model=ActorCritic().to(DEVICE); optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    memory=defaultdict(lambda:{"obs":[],"actions":[],"logp":[],"values":[],"rewards":[],"dones":[]})
    recent=[]; last_loss=0.0
    for match in range(1,matches+1):
        engine=LocalBomberland(seed=match,player_count=4); totals=defaultdict(float)
        while not engine.finished(max_steps):
            state=engine.state(); previous={key:(p.score,p.alive) for key,p in engine.players.items()}
            selected={}
            for key,player in engine.players.items():
                if not player.alive: continue
                obs=encode_observation(state,key); tensor=torch.as_tensor(obs,dtype=torch.float32,device=DEVICE)
                with torch.no_grad(): distribution,value=model(tensor); action=distribution.sample()
                selected[key]=(obs,action.item(),distribution.log_prob(action).item(),value.item())
            engine.step({key:ACTIONS[data[1]] for key,data in selected.items()})
            alive_count=sum(p.alive for p in engine.players.values())
            for key,(obs,action,logp,value) in selected.items():
                player=engine.players[key]; old_score,was_alive=previous[key]
                reward=.01+(player.score-old_score)*.05
                if was_alive and not player.alive: reward-=5.0
                if player.alive and alive_count==1: reward+=10.0
                done=not player.alive or engine.finished(max_steps)
                row=memory[key]; row["obs"].append(obs); row["actions"].append(action); row["logp"].append(logp); row["values"].append(value); row["rewards"].append(reward); row["dones"].append(done)
                totals[key]+=reward
        recent.append(sum(totals.values())/4); recent=recent[-50:]
        if match%update_every==0:
            last_loss=update(model,optimizer,memory); memory.clear()
        if match==1 or match%max(1,matches//20)==0:
            print(f"Match {match:>5}/{matches} | reward {np.mean(recent):>7.2f} | loss {last_loss:>7.3f}")
    if memory: update(model,optimizer,memory)
    torch.save({"model_state":model.state_dict(),"actions":ACTIONS,"matches":matches},output)
    print(f"\nSaved shared PPO policy to {output} ({DEVICE})")


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--matches",type=int,default=2000); parser.add_argument("--output",type=Path,default=Path("bomberland_ppo.pt")); parser.add_argument("--update-every",type=int,default=8); parser.add_argument("--max-steps",type=int,default=300)
    args=parser.parse_args(); train(args.matches,args.output,args.update_every,args.max_steps)
