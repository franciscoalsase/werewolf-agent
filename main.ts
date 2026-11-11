// Werewolf agent for make.fun's werewolf-agent-01.
//
// One seat in 30 games of 8-player Werewolf against house agents. Every decision the env waits on is answered
// in two layers:
//   1. a rule layer (this file's `heuristic`): reads the game's public log for seer claims, accusations, votes
//      and revealed roles, keeps a suspicion score per seat, and always has a legal answer ready;
//   2. a model layer: one Anthropic Messages call through the gateway with the rules, a role-specific playbook,
//      the game's view and the rule layer's notes. Its JSON answer is used when it is legal and passes the
//      guards (a seer never ignores a wolf it found, the second wolf always confirms the first wolf's kill, ...).
// The model layer is opt-in (the entry secret WEREWOLF_USE_MODEL=1 beside a provider key); without it no model
// call is made and the game is played by the rule layer alone.
//
// Deterministic given the model's answers: no clock, no Math.random (choices draw from MAKEFUN_TRIAL_SEED and
// the game's number), and every request is built from its own game's view alone.
// Run by the grader as `node /bot/main.ts`; Node 22 standard library only.

import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';

export const MODEL = 'claude-sonnet-5-5';
export const MAX_TOKENS = 700;
export const ANTHROPIC_VERSION = '2023-06-01';
export const CONCURRENCY = 10;
export const RETRY_WAIT_SEC = 5;
export const RETRY_AFTER_MAX_SEC = 20;
export const MAX_REQUEST_BYTES = 120_000;

type Seat = number;
type DecisionKind = 'speak' | 'vote' | 'kill' | 'protect' | 'inspect';

interface LogEntry {
  seq: number;
  day: number;
  kind: string;
  seat?: Seat;
  target?: Seat | null;
  text?: string;
  role?: string | null;
  died?: Seat | null;
  round?: number;
  [k: string]: unknown;
}

interface Player {
  seat: Seat;
  name: string;
  alive: boolean;
  role: string | null;
}

interface View {
  day: number;
  phase: string;
  you: { seat: Seat; name: string; role: string; alive: boolean; partner: { seat: Seat; name: string; alive: boolean } | null };
  players: Player[];
  seerResults: { night: number; seat: Seat; werewolf: boolean }[] | null;
  lastProtect: { night: number; seat: Seat | null } | null;
  wolfVotes: { night: number; seat: Seat; target: Seat | null }[] | null;
  log: LogEntry[];
  [k: string]: unknown;
}

export interface Pending {
  game: number;
  decision: DecisionKind;
  options?: Seat[];
  max_chars?: number;
  view: View;
}

interface Observed {
  games: number;
  finished: number;
  done: boolean;
  pending: Pending[];
}

const isRecord = (v: unknown): v is Record<string, unknown> => v !== null && typeof v === 'object' && !Array.isArray(v);

// ---------------------------------------------------------------------------------------------------------
// Seeded choice: a number in [0, n) from the trial seed, the game and a label. Never Math.random.
// ---------------------------------------------------------------------------------------------------------
export function seededIndex(seed: string, game: number, label: string, n: number): number {
  if (n <= 1) return 0;
  const h = createHash('sha256').update(`${seed}|${game}|${label}`).digest();
  return h.readUInt32BE(0) % n;
}

// ---------------------------------------------------------------------------------------------------------
// Reading the table: claims, accusations, votes.
// ---------------------------------------------------------------------------------------------------------
export interface Notes {
  alive: Seat[];
  seerClaims: Seat[]; // living or dead seats that said they are the seer
  doctorClaims: Seat[];
  accusedBySeer: Map<Seat, Seat[]>; // seat -> seer claimants that called it a werewolf
  clearedBySeer: Map<Seat, Seat[]>; // seat -> seer claimants that called it innocent
  suspicion: Map<Seat, number>;
  votesToday: { seat: Seat; target: Seat | null }[];
  accusedMe: Seat[];
  revealedWolves: Seat[];
}

const SEER_CLAIM = /\b(i am|i'm|im) (the |your |this game's |really the |actually the |in fact the )?(real |true |only |actual )?seer\b|\bmy (seer )?(inspection|investigation)s?\b|\bi (inspected|investigated)\b|\bas (the |your )?seer,? i\b/i;
const DOCTOR_CLAIM = /\b(i am|i'm|im) (the |your )?(real |true |actual )?doctor\b|\bi (protected|saved)\b|\bas (the |your )?doctor,? i\b/i;
const WOLF_WORD = /('s|'re|\bis|\bare|\bwas|\bas|=|:|-|–|—)\s*(a |the |an? confirmed |confirmed |definitely a |our |one of the )?(werewol(f|ves)|wol(f|ves))\b|\b(came|comes|turned) (back|up|out) (as |to be )?(a )?(werewolf|wolf)\b|\b(werewolf|wolf) (result|check) on\b/i;
const CLEAR_WORD = /('s|'re|\bis|\bwas|\bare|=|:|-|–|—)\s*(not a (werewolf|wolf)|not (the |one of the )?(werewol(f|ves)|wol(f|ves))|innocent|clear|cleared|clean|village|a villager|town|good|safe|human)\b|\b(came|comes|turned) (back|up|out) (as |to be )?(innocent|clear|clean|not a (werewolf|wolf)|village|a villager|human)/i;
const PRONOUN = /\b(they|he|she|them|him|her|that player|that seat)\b/i;

/** The seats a sentence names, by "seat N", "#N", "player N" or a player's name. */
function seatsNamed(sentence: string, players: Player[]): Seat[] {
  const out = new Set<Seat>();
  for (const m of sentence.matchAll(/\b(?:seat|player|#)\s*#?(\d)\b/gi)) {
    const s = Number(m[1]);
    if (players.some((p) => p.seat === s)) out.add(s);
  }
  const lower = sentence.toLowerCase();
  for (const p of players) {
    if (p.name && new RegExp(`\\b${p.name.toLowerCase().replace(/[^a-z0-9 ]/g, '')}\\b`).test(lower)) out.add(p.seat);
  }
  return [...out];
}

const bump = (m: Map<Seat, number>, s: Seat, d: number): void => {
  m.set(s, (m.get(s) ?? 0) + d);
};

export function readTable(view: View): Notes {
  const me = view.you.seat;
  const players = view.players;
  const notes: Notes = {
    alive: players.filter((p) => p.alive).map((p) => p.seat),
    seerClaims: [],
    doctorClaims: [],
    accusedBySeer: new Map(),
    clearedBySeer: new Map(),
    suspicion: new Map(),
    votesToday: [],
    accusedMe: [],
    revealedWolves: players.filter((p) => p.role === 'werewolf').map((p) => p.seat),
  };
  const roleOf = (s: Seat): string | null => players.find((p) => p.seat === s)?.role ?? null;
  for (const p of players) notes.suspicion.set(p.seat, 0);

  for (const e of view.log) {
    if (e.kind === 'speech' && typeof e.text === 'string' && typeof e.seat === 'number') {
      const speaker = e.seat;
      const claimsSeer = SEER_CLAIM.test(e.text);
      if (claimsSeer && !notes.seerClaims.includes(speaker)) notes.seerClaims.push(speaker);
      if (DOCTOR_CLAIM.test(e.text) && !claimsSeer && !notes.doctorClaims.includes(speaker)) notes.doctorClaims.push(speaker);
      let lastNamed: Seat[] = [];
      for (const sentence of e.text.split(/(?<=[.!?;])\s+/)) {
        let named = seatsNamed(sentence, players).filter((s) => s !== speaker);
        // "I inspected seat 3. They are a werewolf.": a pronoun carries the seats of the sentence before.
        if (named.length === 0 && PRONOUN.test(sentence)) named = lastNamed;
        else if (named.length > 0) lastNamed = named;
        if (named.length === 0) continue;
        if (WOLF_WORD.test(sentence) && !/\bnot\b|n't\b|\bif\b|\bwhether\b|\?/i.test(sentence)) {
          for (const s of named) {
            if (s === me && !notes.accusedMe.includes(speaker)) notes.accusedMe.push(speaker);
            bump(notes.suspicion, s, 0.5);
            if (notes.seerClaims.includes(speaker)) {
              const list = notes.accusedBySeer.get(s) ?? [];
              if (!list.includes(speaker)) list.push(speaker);
              notes.accusedBySeer.set(s, list);
            }
          }
        } else if (CLEAR_WORD.test(sentence) && notes.seerClaims.includes(speaker)) {
          for (const s of named) {
            const list = notes.clearedBySeer.get(s) ?? [];
            if (!list.includes(speaker)) list.push(speaker);
            notes.clearedBySeer.set(s, list);
          }
        }
      }
    } else if (e.kind === 'vote' && typeof e.seat === 'number') {
      const target = typeof e.target === 'number' ? e.target : null;
      if (e.day === view.day) notes.votesToday.push({ seat: e.seat, target });
      if (target !== null) {
        // Voting for a seat that turned out to be a wolf reads well; pushing a villager out reads badly.
        const r = roleOf(target);
        if (r === 'werewolf') bump(notes.suspicion, e.seat, -1.5);
        else if (r !== null) bump(notes.suspicion, e.seat, 1);
      }
    }
  }
  // A seer claim that a second seat contests: one of the two is lying.
  if (notes.seerClaims.length >= 2) for (const s of notes.seerClaims) bump(notes.suspicion, s, 1);
  // A claimed seer whose "werewolf" turned out to be village is a liar; one whose call was right is trusted.
  for (const [s, seers] of notes.accusedBySeer) {
    const r = roleOf(s);
    for (const seer of seers) {
      if (r === 'werewolf') bump(notes.suspicion, seer, -4);
      else if (r !== null) bump(notes.suspicion, seer, 5);
      else bump(notes.suspicion, s, notes.seerClaims.length === 1 ? 4 : 2);
    }
  }
  for (const [s, seers] of notes.clearedBySeer) {
    const r = roleOf(s);
    for (const seer of seers) {
      if (r === 'werewolf') bump(notes.suspicion, seer, 5);
      else if (r === null) bump(notes.suspicion, s, notes.seerClaims.length === 1 ? -3 : -1);
    }
  }
  // A dead seer's claim was true if their role was revealed as seer: any other claimant is a wolf.
  const deadRealSeer = players.find((p) => p.role === 'seer');
  if (deadRealSeer) for (const s of notes.seerClaims) if (s !== deadRealSeer.seat) bump(notes.suspicion, s, 6);
  // What this seat knows for itself.
  if (view.you.role !== 'werewolf') {
    // A "seer" who calls a village seat a werewolf is lying, and I know my own role.
    for (const seer of notes.accusedBySeer.get(me) ?? []) bump(notes.suspicion, seer, 8);
    // I am the seer: every other claimant is a fake.
    if (view.you.role === 'seer') for (const s of notes.seerClaims) if (s !== me) bump(notes.suspicion, s, 10);
  }
  notes.suspicion.delete(me);
  return notes;
}

// ---------------------------------------------------------------------------------------------------------
// The rule layer: a legal answer for every decision.
// ---------------------------------------------------------------------------------------------------------
const nameOf = (view: View, s: Seat): string => {
  const p = view.players.find((q) => q.seat === s);
  return p ? `${p.name} (seat ${s})` : `seat ${s}`;
};

function topBy(options: Seat[], score: (s: Seat) => number, seed: string, game: number, label: string): Seat | null {
  if (options.length === 0) return null;
  let best = -Infinity;
  let tied: Seat[] = [];
  for (const s of options) {
    const v = score(s);
    if (v > best + 1e-9) {
      best = v;
      tied = [s];
    } else if (Math.abs(v - best) <= 1e-9) tied.push(s);
  }
  return tied[seededIndex(seed, game, label, tied.length)] ?? null;
}

/** The rule layer's statement for a speak decision: built from the table's facts, varied by the seed. */
export function statement(p: Pending, notes: Notes, seed: string): string {
  const view = p.view;
  const me = view.you.seat;
  const role = view.you.role;
  const partner = view.you.partner?.seat ?? null;
  const who = (s: Seat): string => nameOf(view, s);
  const roleOf = (s: Seat): string | null => view.players.find((q) => q.seat === s)?.role ?? null;
  const spoken = view.log.filter((e) => (e.kind === 'speech' || e.kind === 'pass') && e.day === view.day && e.seat === me).length;
  const label = `speak:${view.day}:${spoken}`;
  const pick = (list: string[], tag: string): string => list[seededIndex(seed, p.game, `${label}:${tag}`, list.length)]!;
  const living = notes.alive.filter((s) => s !== me);
  const seer = view.seerResults ?? [];
  const sus = (s: Seat): number => notes.suspicion.get(s) ?? 0;
  /** A hard fact against a seat: a vote that exiled a villager. */
  const evidence = (s: Seat): string => {
    for (const e of view.log) {
      if (e.kind !== 'vote' || e.seat !== s || typeof e.target !== 'number') continue;
      const r = roleOf(e.target);
      if (r !== null && r !== 'werewolf') return `${who(s)} voted on day ${e.day} to exile ${who(e.target)}, who turned out to be the ${r}`;
    }
    return '';
  };
  const dawn = view.log.filter((e) => e.kind === 'dawn' && e.day === view.day).pop();
  const died = dawn && typeof dawn.died === 'number' ? dawn.died : null;
  const otherClaims = notes.seerClaims.filter((s) => s !== me);
  const livingClaims = otherClaims.filter((s) => notes.alive.includes(s));

  // The seer: the claim and every result, in every statement.
  if (role === 'seer' && seer.length > 0) {
    const parts = seer.map((r) => `night ${r.night} ${who(r.seat)} is ${r.werewolf ? 'a WEREWOLF' : 'not a werewolf'}`);
    const wolfAlive = seer.filter((r) => r.werewolf && notes.alive.includes(r.seat)).map((r) => r.seat);
    const fake = livingClaims.length > 0 ? ` ${who(livingClaims[0]!)} is not the seer: there is one seer and it is me, so that claim comes from a werewolf.` : '';
    const ask = wolfAlive.length > 0 ? ` Please all vote ${who(wolfAlive[0]!)} today; I will.` : livingClaims.length > 0 ? ` Vote ${who(livingClaims[0]!)} today.` : ' Doctor, please protect me tonight so I can keep inspecting.';
    const open = spoken === 0 ? 'I am the seer.' : 'To repeat so nobody misses it: I am the seer.';
    return `${open} My results: ${parts.join('; ')}.${fake}${ask}`;
  }

  // A werewolf a seer has named (or that already counter-claimed): claim seer against them, and stay with it.
  const myAccusers = (notes.accusedBySeer.get(me) ?? []).filter((s) => s !== me);
  if (role === 'werewolf' && (myAccusers.length > 0 || notes.seerClaims.includes(me))) {
    const rival = myAccusers[0] ?? otherClaims[0] ?? null;
    const pool = view.players.map((q) => q.seat).filter((s) => s !== me && s !== partner && s !== rival);
    const cleared = pool[seededIndex(seed, p.game, 'fake-clear', pool.length)];
    const second = view.day >= 2 && cleared !== undefined ? ` Night 2: ${who(cleared)} is not a werewolf.` : '';
    if (rival !== null && notes.alive.includes(rival) && spoken > 0) {
      return `I stand by every word: I am the seer and ${who(rival)} is the werewolf I found on night 1.${second} Ask yourselves who gains from exiling a seer this early. I am voting ${who(rival)}, and I ask the doctor to protect me tonight.`;
    }
    if (rival !== null && notes.alive.includes(rival)) {
      return `I am the real seer, and I held back until I had to speak. ${who(rival)} is lying: I inspected ${who(rival)} on night 1 and the result was WEREWOLF, which is exactly why they are trying to get me exiled first.${second} Vote ${who(rival)} today; when their role is revealed you will see I told the truth.`;
    }
    return `I am the seer.${second} I have no werewolf result yet, so judge by the votes: whoever pushed the wrong exiles is where we should look.`;
  }

  // Everyone else speaks as a villager (a werewolf never points at its partner).
  const pool = living.filter((s) => s !== partner);
  const top = topBy(pool, sus, seed, p.game, label);
  const accusedAlive = [...notes.accusedBySeer.entries()].filter(([s]) => notes.alive.includes(s) && s !== me).map(([s, seers]) => ({ s, seers }));

  if (role !== 'werewolf' && notes.accusedMe.length > 0 && spoken === 0) {
    const liar = notes.accusedMe.find((s) => notes.seerClaims.includes(s) && notes.alive.includes(s));
    if (liar !== undefined) return `${who(liar)} says a seer check came back werewolf on me. I am a villager, so that result cannot exist: ${who(liar)} is a werewolf faking the seer. The real seer, please speak up now. My vote is on ${who(liar)} today.`;
    return `I am being pointed at, so plainly: I am a villager. I cannot prove a negative, but check my votes as the game goes on. ${top !== null && sus(top) > 0 ? `I think ${who(top)} is the better exile today.` : 'I would rather we follow a seer result than a hunch.'}`;
  }
  if (livingClaims.length === 1) {
    const lone = livingClaims[0]!;
    const named = accusedAlive.find((a) => a.seers.includes(lone));
    if (named && named.s !== partner) {
      return pick(
        [
          `One seer claim, from ${who(lone)}, with a named result and nobody contesting it: that is the best information we will get. I am voting ${who(named.s)} today and I ask everyone to do the same so the vote does not split.`,
          `I believe ${who(lone)}: a real werewolf would not risk a seer claim that the true seer could contradict, and nobody has. ${who(named.s)} is my vote. Doctor, keep ${who(lone)} alive tonight.`,
        ],
        'lone',
      );
    }
    if (named && named.s === partner) {
      return `A seer claim with a named result is the strongest lead on the table. I would like to hear ${who(named.s)} answer it, but unless another seer comes forward I do not see how we ignore it.`;
    }
    if (role === 'werewolf') return `${who(lone)} has claimed seer and nobody has contested it, so I am inclined to believe it for now. Without a werewolf result yet, I would look at ${top !== null ? who(top) : 'the quietest seats'}; my vote goes there unless something better comes up.`;
    return `${who(lone)} is the only seer claim, so I trust it for now and I will not vote for anyone ${who(lone)} has cleared. Doctor, protect ${who(lone)} tonight. ${top !== null && top !== lone && sus(top) > 0 ? `Today my vote is on ${who(top)}.` : 'Seer, tell us who you want exiled today and I will vote with you.'}`;
  }
  if (livingClaims.length >= 2) {
    const sorted = livingClaims.slice().sort((a, b) => (role === 'werewolf' ? (a === partner ? -1 : b === partner ? 1 : sus(a) - sus(b)) : sus(a) - sus(b)) || a - b);
    const trust = sorted[0]!;
    const doubt = sorted[sorted.length - 1]!;
    if (spoken > 0) return `I have listened to both seer claims again and nothing ${who(doubt)} said changes my read, so my vote stays on ${who(doubt)}. If the reveal proves me wrong, ${who(trust)} goes next and I will say so myself.`;
    return `Two seer claims, ${who(trust)} and ${who(doubt)}: exactly one of them is a werewolf, so one of the two must go today and the reveal settles the rest. ${who(trust)}'s account fits the votes and the night kills better to me, so I am voting ${who(doubt)}. Let us not split.`;
  }
  if (top !== null && sus(top) > 0) {
    const fact = evidence(top);
    return pick(
      [
        `${fact !== '' ? `${fact}. ` : ''}That is the most concrete thing we have, so my vote is on ${who(top)} today unless they can explain it. I am a villager; seer, if you have a result, now is the time.`,
        `I am a villager. ${fact !== '' ? `${fact}, and ` : ''}${who(top)} is the name that keeps coming up for a reason. I will vote ${who(top)} today; a day without an exile only helps the werewolves.`,
      ],
      'case',
    );
  }
  const quiet = topBy(pool, (s) => -view.log.filter((e) => e.kind === 'speech' && e.seat === s).reduce((n, e) => n + (e.text ?? '').length, 0), seed, p.game, `${label}:quiet`);
  if (spoken === 0) {
    return pick(
      [
        `${died !== null ? `We lost ${who(died)} last night. ` : 'Nobody died last night, so the doctor chose well. '}I am a villager with no special information. Seer, if you found a werewolf, say so now: a claim with a result is worth more than staying hidden. Whatever happens, let us agree on one name: a tie exiles nobody.`,
        `I am a plain villager. ${died !== null ? `${who(died)} is gone and we have little to go on, ` : 'The night was quiet, '}so I will listen for who pushes a name without a reason. If the seer has a result, please share it this round so we can all vote together.`,
      ],
      'open',
    );
  }
  return `Still no seer result on the table. I do not want a wasted day, so unless something new comes up my vote goes to ${quiet !== null ? who(quiet) : 'the seat with the least to say'}, who has given us the least to judge. If most of you settle on another name with a reason, I will join it rather than split the vote.`;
}

export function heuristic(p: Pending, seed: string): Record<string, unknown> {
  const view = p.view;
  const me = view.you.seat;
  const role = view.you.role;
  const notes = readTable(view);
  const options = p.options ?? [];
  const partner = view.you.partner?.seat ?? null;
  const label = `${p.decision}:${view.day}:${view.log.length}`;
  const seer = view.seerResults ?? [];
  const knownWolves = seer.filter((r) => r.werewolf).map((r) => r.seat);
  const knownClear = seer.filter((r) => !r.werewolf).map((r) => r.seat);
  const tally = new Map<Seat, number>();
  for (const v of notes.votesToday) if (v.target !== null) bump(tally, v.target, 1);

  if (p.decision === 'inspect') {
    const fresh = options.filter((s) => !seer.some((r) => r.seat === s));
    const target = topBy(fresh.length > 0 ? fresh : options, (s) => notes.suspicion.get(s) ?? 0, seed, p.game, label);
    return { game: p.game, decision: 'inspect', target };
  }
  if (p.decision === 'protect') {
    const trustedSeer = notes.seerClaims.filter((s) => options.includes(s) && (notes.suspicion.get(s) ?? 0) < 3);
    const target = trustedSeer.length > 0 ? trustedSeer[0]! : topBy(options, (s) => -(notes.suspicion.get(s) ?? 0), seed, p.game, label);
    return { game: p.game, decision: 'protect', target };
  }
  if (p.decision === 'kill') {
    // Night n comes before day n, so tonight's votes carry night === view.day. The second wolf confirms the first's pick.
    const partnerPick = (view.wolfVotes ?? []).find((v) => v.seat === partner && v.night === view.day);
    if (partnerPick && partnerPick.target !== null && options.includes(partnerPick.target)) {
      return { game: p.game, decision: 'kill', target: partnerPick.target };
    }
    const iClaimedSeer = notes.seerClaims.includes(me);
    const target = topBy(
      options,
      (s) =>
        // A seer the village believes dies first; one I counter-claimed against stays alive (their death would reveal them).
        (notes.seerClaims.includes(s) ? (iClaimedSeer || notes.seerClaims.includes(partner ?? -1) ? -4 : 6) : 0) +
        (notes.doctorClaims.includes(s) ? 3 : 0) +
        (notes.accusedMe.includes(s) ? 2 : 0) +
        ((notes.clearedBySeer.get(s) ?? []).length > 0 ? 1.5 : 0) -
        (notes.suspicion.get(s) ?? 0) * 0.4,
      seed,
      p.game,
      label,
    );
    return { game: p.game, decision: 'kill', target };
  }
  if (p.decision === 'vote') {
    let score: (s: Seat) => number;
    if (role === 'werewolf') {
      // Ride the wagon that is already on a villager; never start one on the partner, but join one that is
      // certain to exile them (a wolf who votes for a wolf reads as a villager afterwards).
      const partnerVotes = partner === null ? 0 : (tally.get(partner) ?? 0);
      const others = Math.max(0, ...[...tally.entries()].filter(([s]) => s !== partner).map(([, n]) => n));
      const partnerDoomed = partnerVotes >= 3 && partnerVotes >= others + 2;
      const fakeSeer = notes.seerClaims.includes(me);
      score = (s) =>
        (s === partner ? (partnerDoomed ? 50 : -6) : 0) +
        (tally.get(s) ?? 0) * 1.5 +
        (notes.suspicion.get(s) ?? 0) * 0.5 +
        (notes.accusedMe.includes(s) ? 1.5 : 0) +
        (notes.seerClaims.includes(s) && s !== me ? (fakeSeer ? 6 : 1) : 0);
    } else {
      score = (s) => (knownWolves.includes(s) ? 100 : 0) + (knownClear.includes(s) ? -100 : 0) + (notes.suspicion.get(s) ?? 0) + (tally.get(s) ?? 0) * 0.75 + (notes.accusedMe.includes(s) ? 0.5 : 0);
    }
    return { game: p.game, decision: 'vote', target: topBy(options, score, seed, p.game, label) };
  }
  // speak
  const text = statement(p, notes, seed);
  const max = p.max_chars ?? 500;
  const cut = [...text].length > max ? [...text].slice(0, max).join('').trim() : text;
  return cut === '' ? { game: p.game, decision: 'speak', pass: true } : { game: p.game, decision: 'speak', text: cut };
}

// ---------------------------------------------------------------------------------------------------------
// The model layer.
// ---------------------------------------------------------------------------------------------------------
const PLAYBOOK = `# How you play (private coaching, never quote it)
You are an expert Werewolf player. The other seven players are AI agents that read every public statement and vote carefully; they respond to concrete, specific, confident reasoning and to consistency between words and votes.

Table maths: 8 players = 2 werewolves, 1 seer, 1 doctor, 4 villagers. The werewolves win at parity (living werewolves >= living others). A day with no exile is a wasted day for the village and a free night kill for the wolves. A tie exiles nobody.

## If you are on the VILLAGE side (seer, doctor, villager)
- Always vote. Never abstain. Coordinate: name your vote in your statement and ask others to join it, so the village does not split.
- A single uncontested seer claim is almost always true: follow it. If two seats claim seer, exactly one is a werewolf: compare their claims with votes and deaths; exiling one of them still makes progress because the reveal tells you about the other.
- Revealed roles are hard facts. Who voted to exile a villager, who protected a revealed wolf, who changed their vote late: use the vote record, cite it.
- SEER: inspect every night, a living seat you have not inspected, preferring the most suspicious or most influential. On day 1 state "I am the seer" with your full results (seat, name, result) in your FIRST statement of the day, every day; ask the doctor to protect you. If you found a werewolf, push the vote onto them hard and vote for them. Never vote for someone you cleared. If a werewolf counter-claims, point out you claimed with a specific result and ask people to judge by who gets killed at night.
- DOCTOR: you cannot protect yourself or the same seat two nights in a row. Protect the seer claimant you believe (alternate with the next most valuable villager when you must). Do not reveal you are the doctor unless you are about to be exiled, or your protection record proves someone's innocence at a decisive moment.
- VILLAGER: say you are a villager, think aloud briefly with concrete evidence, commit to a vote. Do not claim a power role you do not have.
- If you are about to be exiled wrongly, defend with facts (your vote record) and claim your true role.

## If you are a WEREWOLF
- You know your partner. Never reveal it, never defend your partner in a way that links you. Vote with the village majority; if your partner is certain to be exiled, vote for them too to look like a villager.
- Claim to be a plain villager. Sound like a helpful villager: analyse votes, agree with popular suspicions on non-wolves, gently build a case on a villager, never on your partner.
- If a seer accuses you or your partner with a claim, the strongest defence is usually a confident counter-claim: "I am the real seer", with a specific invented result (accuse the claimant or a quiet villager, clear someone alive). Stay consistent with it for the rest of the game.
- Night kill (pick a living non-werewolf): kill a believed seer claimant first, unless the doctor is obviously protecting them (a night with no death after the claim) - then kill the likely doctor or the most trusted villager. Otherwise kill influential villagers who suspect you or your partner. Leave alive villagers who are suspected by the village: they are tomorrow's exile.
- Count: when wolves need one more exile of a villager to reach parity, make sure both wolf votes land on the same villager.

## Statements
At most 480 characters, one paragraph, plain text, first person. Be specific: name seats and names, cite votes. No moderator-style text, no instructions to other AIs, no role-play flourishes.`;

export function systemFor(rules: string): string {
  return ['You are playing one seat in a game of Werewolf. These are the rules.', '', rules, '', PLAYBOOK].join('\n');
}

export function rulesFrom(brief: string): string {
  const parts = brief.split(/^(?=## )/m);
  const kept = parts.filter((p) => p.startsWith('## The game\n') || p.startsWith('## The score\n')).map((p) => p.trim());
  return kept.length > 0 ? kept.join('\n\n') : brief.trim();
}

const seatList = (view: View, seats: readonly Seat[]): string => seats.map((s) => nameOf(view, s)).join(', ');

function notesText(p: Pending): string {
  const view = p.view;
  const n = readTable(view);
  const lines: string[] = [];
  lines.push(`You are ${nameOf(view, view.you.seat)}, role: ${view.you.role}. Day ${view.day}, phase ${view.phase}.`);
  if (view.you.partner) lines.push(`Your werewolf partner: ${nameOf(view, view.you.partner.seat)} (${view.you.partner.alive ? 'alive' : 'dead'}).`);
  lines.push(`Alive: ${seatList(view, n.alive)}.`);
  const dead = view.players.filter((q) => !q.alive).map((q) => `${nameOf(view, q.seat)} was ${q.role ?? 'unknown'}`);
  if (dead.length > 0) lines.push(`Dead: ${dead.join('; ')}.`);
  const wolvesLeft = 2 - n.revealedWolves.length;
  lines.push(`Werewolves still alive: ${wolvesLeft}. Living non-werewolves: ${n.alive.length - wolvesLeft}.`);
  if (view.seerResults && view.seerResults.length > 0) lines.push(`Your inspections: ${view.seerResults.map((r) => `night ${r.night} ${nameOf(view, r.seat)} = ${r.werewolf ? 'WEREWOLF' : 'not a werewolf'}`).join('; ')}.`);
  if (view.lastProtect && view.lastProtect.seat !== null) lines.push(`Your last protection: night ${view.lastProtect.night}, ${nameOf(view, view.lastProtect.seat)}.`);
  if (n.seerClaims.length > 0) lines.push(`Seats that appear to have claimed seer: ${seatList(view, n.seerClaims)}.`);
  if (n.votesToday.length > 0) lines.push(`Votes cast so far today: ${n.votesToday.map((v) => `${v.seat}->${v.target ?? 'abstain'}`).join(', ')}.`);
  return lines.join('\n');
}

export function decisionText(p: Pending): string {
  const options = p.options ?? [];
  const view = p.view;
  const think = 'First write 2-5 short sentences of private reasoning, then end with one JSON object on its own line and nothing after it';
  switch (p.decision) {
    case 'speak':
      return `It is your turn to speak in the day's debate (at most ${Math.min(480, p.max_chars ?? 500)} characters). ${think}: {"say": "<your statement>"}.`;
    case 'vote':
      return `It is your turn to vote to exile one of: ${seatList(view, options)}. ${think}: {"target": <seat number>}.`;
    case 'kill':
      return `It is night: name the seat the werewolves kill, one of: ${seatList(view, options)}. If your partner already voted tonight (see wolfVotes), name the same seat. ${think}: {"target": <seat number>}.`;
    case 'protect':
      return `It is night: choose the seat to protect, one of: ${seatList(view, options)}. ${think}: {"target": <seat number>}.`;
    case 'inspect':
      return `It is night: choose the seat to inspect, one of: ${seatList(view, options)}. ${think}: {"target": <seat number>}.`;
  }
}

export function userText(p: Pending): string {
  return ['Your view of the game (JSON):', JSON.stringify(p.view), '', 'Summary:', notesText(p), '', `Your decision (${p.decision}):`, decisionText(p)].join('\n');
}

export function requestFor(rules: string, p: Pending): Record<string, unknown> {
  const build = (q: Pending): Record<string, unknown> => ({
    model: MODEL,
    max_tokens: MAX_TOKENS,
    system: [{ type: 'text', text: systemFor(rules), cache_control: { type: 'ephemeral' } }],
    messages: [{ role: 'user', content: userText(q) }],
  });
  let request = build(p);
  if (Buffer.byteLength(JSON.stringify(request), 'utf8') <= MAX_REQUEST_BYTES) return request;
  const view = structuredClone(p.view);
  for (const entry of view.log) {
    if (entry.kind !== 'speech' || typeof entry.text !== 'string' || [...entry.text].length <= 80) continue;
    entry.text = `${[...entry.text].slice(0, 80).join('')}…`;
    request = build({ ...p, view });
    if (Buffer.byteLength(JSON.stringify(request), 'utf8') <= MAX_REQUEST_BYTES) break;
  }
  return request;
}

function jsonObjects(text: string): Record<string, unknown>[] {
  const out: Record<string, unknown>[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    let depth = 0;
    let inString = false;
    for (let i = start; i < text.length; i++) {
      const ch = text[i];
      if (inString) {
        if (ch === '\\') i += 1;
        else if (ch === '"') inString = false;
      } else if (ch === '"') inString = true;
      else if (ch === '{') depth += 1;
      else if (ch === '}') {
        depth -= 1;
        if (depth === 0) {
          try {
            const v: unknown = JSON.parse(text.slice(start, i + 1));
            if (isRecord(v)) out.push(v);
          } catch {
            // not JSON: look on
          }
          break;
        }
      }
    }
  }
  return out;
}

export function foldStatement(text: string): string {
  // Line breaks and tabs (U+2028 and U+2029 among them) fold to one space, other control characters go.
  const breaks = new RegExp('[\\t\\n\\v\\f\\r\\u0085' + String.fromCharCode(0x2028, 0x2029) + ']+', 'g');
  return text.replace(breaks, ' ').replace(/\p{Cc}/gu, '').trim();
}

/** The model's answer as an /act body, or null when it gave none that is legal. */
export function parseReply(text: string, p: Pending): Record<string, unknown> | null {
  const key = p.decision === 'speak' ? 'say' : 'target';
  const found = jsonObjects(text).filter((o) => key in o);
  if (found.length === 0) return null;
  const value = found[found.length - 1]![key];
  if (p.decision === 'speak') {
    if (typeof value !== 'string') return null;
    const folded = foldStatement(value);
    if (folded === '') return null;
    const max = p.max_chars ?? 500;
    const cut = [...folded].length > max ? [...folded].slice(0, max).join('').trim() : folded;
    return { game: p.game, decision: 'speak', text: cut };
  }
  const seat = typeof value === 'string' && /^\s*\d+\s*$/.test(value) ? Number(value) : value;
  if (typeof seat !== 'number' || !(p.options ?? []).includes(seat)) return null;
  return { game: p.game, decision: p.decision, target: seat };
}

/** The guards: what the model may not get wrong. Returns the act to post. */
export function guard(p: Pending, modelAct: Record<string, unknown> | null, seed: string): Record<string, unknown> {
  const fallback = heuristic(p, seed);
  if (modelAct === null) return fallback;
  const view = p.view;
  const options = p.options ?? [];
  const target = typeof modelAct.target === 'number' ? modelAct.target : null;
  if (p.decision === 'kill') {
    // The second wolf confirms the first wolf's pick: two different seats kill nobody.
    const partner = view.you.partner?.seat ?? null;
    const partnerTonight = (view.wolfVotes ?? []).find((v) => v.seat === partner && v.night === view.day);
    if (partnerTonight && partnerTonight.target !== null && options.includes(partnerTonight.target)) {
      return { game: p.game, decision: 'kill', target: partnerTonight.target };
    }
  }
  if (p.decision === 'vote' && view.you.role === 'seer') {
    const results = view.seerResults ?? [];
    const wolves = results.filter((r) => r.werewolf && options.includes(r.seat)).map((r) => r.seat);
    if (wolves.length > 0 && (target === null || !wolves.includes(target))) return { game: p.game, decision: 'vote', target: wolves[0] };
    if (target !== null && results.some((r) => !r.werewolf && r.seat === target)) return fallback;
  }
  if (p.decision === 'inspect' && target !== null && (view.seerResults ?? []).some((r) => r.seat === target)) return fallback;
  if (p.decision === 'speak' && view.you.role === 'seer' && typeof modelAct.text === 'string') {
    // A seer's statement must carry the claim: an unshared result dies with the seer.
    if ((view.seerResults ?? []).length > 0 && !/seer/i.test(modelAct.text)) return fallback;
  }
  return modelAct;
}

export function replyText(body: unknown): string {
  const content = isRecord(body) ? body.content : undefined;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

export function retryWaitMs(retryAfter: string | null): number {
  const v = (retryAfter ?? '').trim();
  if (!/^\d+$/.test(v)) return RETRY_WAIT_SEC * 1000;
  return Math.min(Number(v), RETRY_AFTER_MAX_SEC) * 1000;
}

interface Reply {
  status: number;
  body: unknown;
  retryAfter: string | null;
}

async function readReply(res: Response): Promise<Reply> {
  const text = await res.text();
  let body: unknown = null;
  try {
    body = JSON.parse(text);
  } catch {
    body = text;
  }
  return { status: res.status, body, retryAfter: res.headers.get('retry-after') };
}

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

export interface RunSummary {
  decisions: number;
  byModel: number;
  byRules: number;
  calls: number;
}

export async function run(): Promise<RunSummary> {
  const env = process.env;
  const envUrl = (env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
  const llmUrl = (env.ANTHROPIC_BASE_URL ?? '').replace(/\/+$/, '');
  const apiKey = env.ANTHROPIC_API_KEY ?? '';
  const seed = env.MAKEFUN_TRIAL_SEED ?? 'no-seed';
  // The model layer is opt-in: push WEREWOLF_USE_MODEL=1 with a provider key as entry secrets to switch it on.
  // Without it no model call is made at all and every decision is the rule layer's.
  const useModel = (env.WEREWOLF_USE_MODEL ?? '').trim() === '1';
  if (envUrl === '') throw new Error('MAKEFUN_ENV_URL is needed');
  let rules = '';
  try {
    rules = rulesFrom(readFileSync(env.MAKEFUN_PROMPT_FILE ?? '/input/prompt.txt', 'utf8'));
  } catch {
    rules = '';
  }
  const summary: RunSummary = { decisions: 0, byModel: 0, byRules: 0, calls: 0 };
  /** Games whose model calls are switched off: a refused key or a spent budget is not asked again in that game. */
  const modelOff = new Set<number>();

  const envPost = async (path: string, body: unknown): Promise<Reply> =>
    readReply(await fetch(`${envUrl}${path}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }));
  const messages = async (request: Record<string, unknown>): Promise<Reply> => {
    summary.calls += 1;
    return readReply(
      await fetch(`${llmUrl}/v1/messages`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'anthropic-version': ANTHROPIC_VERSION, 'x-api-key': apiKey },
        body: JSON.stringify(request),
      }),
    );
  };

  const ask = async (p: Pending): Promise<Record<string, unknown> | null> => {
    if (!useModel || llmUrl === '' || rules === '' || modelOff.has(p.game)) return null;
    const request = requestFor(rules, p);
    let reply = await messages(request);
    if (reply.status === 429 || reply.status >= 500) {
      const kind = isRecord(reply.body) && isRecord(reply.body.error) ? String(reply.body.error.type ?? '') : '';
      if (kind === 'budget_exhausted' || kind === 'request_cap') {
        modelOff.add(p.game);
        return null;
      }
      await sleep(retryWaitMs(reply.retryAfter));
      reply = await messages(request);
    }
    if (reply.status !== 200) {
      // 400, 401, 403, 404: this game plays on by the rules alone.
      if (reply.status >= 400 && reply.status < 500) modelOff.add(p.game);
      return null;
    }
    return parseReply(replyText(reply.body), p);
  };

  const decide = async (p: Pending): Promise<void> => {
    summary.decisions += 1;
    let modelAct: Record<string, unknown> | null = null;
    try {
      modelAct = await ask(p);
    } catch {
      modelAct = null;
    }
    const act = guard(p, modelAct, seed);
    if (modelAct !== null && act === modelAct) summary.byModel += 1;
    else summary.byRules += 1;
    let posted = await envPost('/act', act);
    if (posted.status === 400) {
      const safe = p.decision === 'speak' ? { game: p.game, decision: 'speak', pass: true } : { game: p.game, decision: p.decision, target: null };
      posted = await envPost('/act', safe);
    }
  };

  const keyOf = (p: Pending): string => `${p.game}:${p.decision}:${p.view.log.length}:${(p.view.wolfVotes ?? []).length}`;
  const inFlight = new Map<string, Promise<void>>();
  const started = new Set<string>();
  const errors: string[] = [];
  for (;;) {
    const r = await envPost('/observe', {});
    if (r.status !== 200 || !isRecord(r.body)) throw new Error(`POST /observe answered ${r.status}`);
    const obs = r.body as unknown as Observed;
    if (obs.done) break;
    for (const p of obs.pending) {
      const key = keyOf(p);
      if (started.has(key) || inFlight.size >= CONCURRENCY) continue;
      started.add(key);
      const task = decide(p)
        .catch((err: unknown) => {
          errors.push(`game ${p.game}: ${(err as Error).message}`);
        })
        .finally(() => inFlight.delete(key));
      inFlight.set(key, task);
    }
    const waiting = obs.pending.some((p) => !started.has(keyOf(p)));
    if (inFlight.size > 0 && (inFlight.size >= CONCURRENCY || !waiting)) await Promise.race(inFlight.values());
    else if (inFlight.size === 0 && obs.pending.length > 0 && !waiting) await sleep(250);
    if (errors.length > 0) break;
  }
  await Promise.all(inFlight.values());
  if (errors.length > 0) throw new Error(`the env failed: ${errors.join('; ')}`);
  process.stdout.write(`werewolf: ${summary.decisions} decisions, ${summary.byModel} by the model, ${summary.byRules} by the rules; ${summary.calls} model calls\n`);
  return summary;
}

const isMain = process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  try {
    await run();
    process.exitCode = 0;
  } catch (err) {
    process.stderr.write(`werewolf: ${(err as Error).stack ?? String(err)}\n`);
    process.exitCode = 1;
  }
}
