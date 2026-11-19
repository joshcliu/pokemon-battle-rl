// Select rosters before looking at battle outcomes. Run against the pinned benchmark simulator.
const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');
const root = path.resolve(__dirname, '..');
const cache = '.cache/vgcbench-external';
const {Teams, TeamValidator, Dex} = require(path.join(root, cache, 'showdown/dist/sim'));
const bank = cache + '/source/teams/reg_mb';
const hash = s => createHash('sha256').update(s).digest('hex');
const team = 'teams/reg-mb-variant/MC461-MB.txt';
const dev = require('../results/sketch5-vgcbench-bc/protocol.json');
const guard = [];
const hashes = {};
const validator = new TeamValidator(dev.format);
function roster(file) {
  const raw = fs.readFileSync(path.join(root, file), 'utf8');
  const sets = Teams.import(raw);
  const species = sets.map(s => Dex.species.get(s.species).baseSpecies);
  return {raw, sets, species};
}
for (const p of [team, ...dev.teams, ...dev.training_teams]) guard.push(roster(p).species);
const candidates = fs.readdirSync(path.join(root, bank)).filter(f => /^MB\d+\.txt$/.test(f));
candidates.sort((a,b) => hash('mb-actionpilot-v1/' + a).localeCompare(hash('mb-actionpilot-v1/' + b)));
const fresh = [];
for (const f of candidates) {
  const p = bank + '/' + f;
  const {raw, sets, species} = roster(p);
  if (sets.length !== 6 || sets.some(s => s.moves.length !== 4 || !s.ability || !s.nature)) continue;
  if (guard.some(other => other.filter(s => species.includes(s)).length > 3)) continue;
  if (validator.validateTeam(Teams.import(raw))?.length) continue;
  guard.push(species); fresh.push(p); hashes[p] = hash(raw);
  if (fresh.length === 23) break;
}
if (fresh.length !== 23) throw Error('Insufficient eligible rosters');
hashes[team] = hash(roster(team).raw);
if (validator.validateTeam(Teams.import(roster(team).raw))?.length) throw Error('Illegal MC461');
const spec = {source:dev.source_revision, simulator:dev.evaluation_simulator, format:dev.format,
  own_train:[team,...fresh.slice(0,3)], foe_train:fresh.slice(3,11),
  own_test:fresh.slice(11,15), foe_test:fresh.slice(15,23), hashes,
  training_seeds:[2,3,4,5], steps:4096, preview_groups:32, group_size:6, battle_seeds_per_group:2,
  eval_games_per_pair:8, development_games_per_team:64, primary:'Sampled win rate on the unseen own / unseen foe cell; average the four training seeds. Opponent is the frozen released BC policy with native preview.'};
const out = path.join(root, 'results/sketch5-mb-posttraining-pilot/protocol.json');
fs.mkdirSync(path.dirname(out), {recursive:true});
const bytes = JSON.stringify(spec,null,2) + '\n';
if (fs.existsSync(out) && fs.readFileSync(out,'utf8') !== bytes) throw Error('Protocol changed');
fs.writeFileSync(out, bytes);
