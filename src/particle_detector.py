from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import hashlib
from typing import Dict, FrozenSet, List, Set

import numpy as np

from src.utils.graph_algorithms import find_connected_clusters
from src.kinematics import calculate_kinematics

# ------------------------------------------------------------
def _path_sig(tag:int, preds:List[bytes]) -> bytes:
    h=hashlib.blake2b(digest_size=16)
    h.update(tag.to_bytes(4,"little"))
    for s in sorted(preds): h.update(s)
    return h.digest()

@dataclass
class Particle:
    id:int; period:int; nodes:FrozenSet[int]
    first_tick:int; last_tick:int; kinematics:dict
    @property
    def lifetime(self): return self.last_tick-self.first_tick

# ------------------------------------------------------------
class ParticleDetector:
    def __init__(self, site, state_mgr, cfg):
        self.site=site; self.state_mgr=state_mgr
        d=cfg.get("detector",{})
        self.max_hist=d.get("max_history_length",10000)
        self.min_period=d.get("min_loop_period",3)
        self.min_size=d.get("min_particle_size",2)

        self._hist={n:deque(maxlen=self.max_hist) for n in site.graph.nodes}
        self._sig_cache={}
        self._clust2pid:Dict[FrozenSet[int],int]={}
        self._next_pid=0
        self.active:Dict[int,Particle]={}
        self.archive:Dict[int,Particle]={}
        self.looping_nodes_last_tick=set()

    # --------------------------------------------------------
    def detect(self,state:np.ndarray,tick:int)->Dict[int,Particle]:
        nodes=np.arange(state.shape[0])
        hidden=self.state_mgr.hidden_nodes
        vis=nodes[~np.isin(nodes,list(hidden))]

        cur={}
        for n in vis:
            preds=[self._sig_cache.get(p,b"\0"*16) for p in self.site.get_predecessors(n)]
            cur[n]=_path_sig(int(state[n]),preds)
        self._sig_cache=cur

        loops:Dict[int,Set[int]]={}
        for n in vis:
            h=self._hist[n]
            sig=cur[n]
            if sig in h:
                p=len(h)-1-h.index(sig)
                if p>=self.min_period: loops.setdefault(p,set()).add(n)
            h.append(sig)

        self.looping_nodes_last_tick={m for s in loops.values() for m in s}

        alive=set()
        for period,nodes_set in loops.items():
            for clust in find_connected_clusters(nodes_set,self.site):
                if len(clust)<self.min_size: continue
                f=frozenset(clust); alive.add(f)

                last=self.active[self._clust2pid[f]].kinematics if f in self._clust2pid else None
                kin=calculate_kinematics(
                        Particle(0,0,f,0,0,{}),
                        self.site.atlas,
                        self.site.metric,
                        last)

                if f in self._clust2pid:
                    pid=self._clust2pid[f]
                    p=self.active[pid]; p.last_tick=tick; p.kinematics=kin
                else:
                    pid=self._next_pid; self._next_pid+=1
                    self._clust2pid[f]=pid
                    self.active[pid]=Particle(pid,period,f,tick,tick,kin)

        # retire
        for dead in set(self._clust2pid)-alive:
            pid=self._clust2pid.pop(dead)
            self.archive[pid]=self.active.pop(pid)

        return self.active
