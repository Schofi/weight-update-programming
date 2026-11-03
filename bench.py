"""Paired, deterministic diagnostic tasks. Only prompts and one probe reach the editor."""
from dataclasses import dataclass
from itertools import product

import torch


DOMAINS = {
    "discovery": [("ticket lookup", "timeout", "retry"), ("hotel lookup", "stale", "repeat")],
    "validation": [("train inventory", "unavailable", "requery")],
    "test": [("rail exchange", "expired", "resubmit"), ("luggage transfer", "pending", "poll"),
             ("coach reservation", "blocked", "replay")],
}
SYSTEM = "You diagnose damaged policies. Follow the requested output format."


@dataclass(frozen=True)
class Case:
    split: str
    domain: int
    fault: int
    good_slot: int
    repair_slot: int

    @property
    def key(self):
        return f"{self.split}-{self.domain}-{self.fault}-{self.good_slot}-{self.repair_slot}"

    @property
    def fields(self):
        _, first, second = DOMAINS[self.split][self.domain]
        return first, second

    def healthy(self):
        # Two action logits: advance vs stay. The third input is nuisance context.
        return torch.tensor([[1., 1., 0.], [0., 0., 0.]])

    def damaged(self):
        w = self.healthy()
        w[0, self.fault] += self.damage
        return w

    @property
    def damage(self):
        # The inverted fault family occurs only on the last held-out domain.
        return 3. if self.split == "test" and self.domain == 2 else -3.

    @property
    def incident(self):
        return [0.5, 0.5, 0.] if self.damage > 0 else [1., 1., 0.]

    def fault_description(self):
        return f"Exactly one of the first two coefficients changed from +1 to {1+self.damage:g}."

    def probes(self):
        isolate = [0., 0., 0.]
        isolate[self.domain % 2] = 1.
        decoy = [1., 1., -1.]
        return (isolate, decoy) if self.good_slot == 0 else (decoy, isolate)

    def repair_columns(self):
        return (0, 1) if self.repair_slot == 0 else (1, 0)

    def patched(self, choice):
        # The chosen repair is an actual rank-one edit to a new target copy.
        u = torch.tensor([-self.damage, 0.])
        v = torch.eye(3)[self.repair_columns()[choice]]
        return self.damaged() + torch.outer(u, v)

    def probe_prompt(self):
        name, first, second = DOMAINS[self.split][self.domain]
        intro = {
            "discovery": f"Diagnose the {name} incident.",
            "validation": f"The {name} workflow chose the wrong next action.",
            "test": f"Incident report from {name}. Identify an informative replay.",
        }[self.split]
        probes = self.probes()
        observed = (self.damaged() @ torch.tensor(self.incident))[0].item()
        return (f"{intro} Inputs in order: [{first}, {second}, style]. "
                f"The healthy advance score is x1+x2. {self.fault_description()} Style never affects the score. "
                f"At {self.incident} the observed advance score is {observed:g}. "
                f"Pick ONE input that distinguishes the two possible faults. "
                f"A: {probes[0]}  B: {probes[1]}. Reply with A or B.")

    def repair_prompt(self, probe, observed):
        fields = self.fields
        a, b = self.repair_columns()
        return (f"Input order: [{fields[0]}, {fields[1]}, style]. Healthy score: x1+x2. "
                f"{self.fault_description()} "
                f"Observed advance score {observed:g} at {self.probes()[probe]}. "
                f"Submit ONE repair. A: add {-self.damage:g} to the {fields[a]} coefficient. "
                f"B: add {-self.damage:g} to the {fields[b]} coefficient. Reply with A or B.")


def cases(split):
    return [Case(split, *args) for args in product(range(len(DOMAINS[split])), range(2),
                                                    range(2), range(2))]


def probe_score(case, probe):
    return (case.damaged() @ torch.tensor(case.probes()[probe]))[0].item()


def grade(case, choice):
    # Both negative and positive signed observations appear. A tie selects advance.
    levels = (-2., -1., 0., 0.5, 1., 2.)
    xs = torch.tensor(list(product(levels, levels, (-1., 1.))))
    # A fixed alternative-action logit makes a wrong repair capable of regressions.
    offset = torch.tensor([0., 1.5])
    desired = (xs @ case.healthy().T + offset).argmax(-1)
    before = (xs @ case.damaged().T + offset).argmax(-1) == desired
    after = (xs @ case.patched(choice).T + offset).argmax(-1) == desired
    return {"repaired": bool(after[~before].all()), "preserved": bool(after[before].all()),
            "joint": bool(after.all()), "correct_column": case.repair_columns()[choice] == case.fault,
            "broken_states": int((~before).sum()), "restored_states": int(after[~before].sum()),
            "intact_states": int(before.sum()), "regressed_states": int((~after[before]).sum())}
