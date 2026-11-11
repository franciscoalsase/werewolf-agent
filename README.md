# werewolf-agent

An entry for make.fun's bounty "Build the best Werewolf agent" (`werewolf-agent-01`).

The agent holds one seat in 30 games of 8-player Werewolf against house agents. Each decision is answered by a
rule layer that reads the public log (seer claims, accusations, votes, revealed roles) and keeps a suspicion
score per seat. An optional model layer (off by default) can be switched on with the entry secret
`WEREWOLF_USE_MODEL=1` beside a provider key.

- Seer: inspects the most suspicious unchecked seat, claims on day 1 with every result, votes its wolf.
- Doctor: protects the believed seer claimant, otherwise the least suspicious seat.
- Villager: follows an uncontested seer, otherwise the vote record.
- Werewolf: confirms its partner's kill, rides the village's wagon, counter-claims seer when named.

Run: `MAKEFUN_ENV_URL=http://127.0.0.1:8080 MAKEFUN_TRIAL_SEED=seed node main.ts` (Node 22, standard library only).

Deterministic: no clock, no unseeded randomness.

License: MIT.
