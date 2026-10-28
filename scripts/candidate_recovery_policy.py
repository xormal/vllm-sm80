"""Pure recovery decisions, separate from systemd and HTTP side effects."""
from dataclasses import dataclass


@dataclass
class RecoveryPolicy:
    invocation: str = ''
    healthy_once: bool = False
    health_failures: int = 0
    counters: tuple | None = None
    progress_at: float = 0
    action_at: float = -1e30
    failed_since: float | None = None

    def decide(self, *, now, invocation, state, age, healthy, metrics, available):
        if state=='failed':
            if self.failed_since is None:self.failed_since=now
        else:self.failed_since=None
        if invocation!=self.invocation:
            self.invocation=invocation
            self.healthy_once=False
            self.health_failures=0
            self.counters=None
            self.progress_at=now
        if healthy:
            self.healthy_once=True
            self.health_failures=0
        elif state=='active' and (self.healthy_once or age>=900):
            self.health_failures+=1
        inflight=(metrics or {}).get('num_requests_running',0)+(metrics or {}).get('num_requests_waiting',0)
        counters=tuple((metrics or {}).get(k,0) for k in
            ('request_success_total','prompt_tokens_total','generation_tokens_total'))
        if not inflight or counters!=self.counters:
            self.progress_at=now
        self.counters=counters
        if available<16*1024**3 or now-self.action_at<300:return None
        action=None
        if state=='failed' and now-self.failed_since>=300:
            action=('start','failed unit persisted for300s; retry after systemd start limit')
        elif state=='active' and self.health_failures>=6:
            action=('restart','six consecutive health failures after startup grace')
        elif state=='active' and healthy and inflight and now-self.progress_at>=600:
            action=('restart','inflight requests without counter progress for600s')
        if action:
            self.action_at=now
            self.health_failures=0
            self.progress_at=now
        return action
