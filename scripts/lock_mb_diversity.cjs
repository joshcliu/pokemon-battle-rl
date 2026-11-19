// Choose extra learner teams from the pinned Reg M-B bank before playing any battles.
const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');
const root = path.resolve(__dirname, '..');
const base = require('../results/sketch5-mb-reward-factorial/protocol.json');
const earlier = require('../results/sketch5-vgcbench-bc/protocol.json');
const {Teams, TeamValidator, Dex} = require(path.join(root, '.cache/vgcbench-external/showdown/dist/sim'));
const bank = '.cache/vgcbench-external/source/teams/reg_mb';
const sha = data => createHash('sha256').update(data).digest('hex');
const validator = new TeamValidator(base.format);

function sheet(file) {
  const bytes = fs.readFileSync(path.join(root, file));
  const sets = Teams.import(bytes.toString());
  const species = sets.map(s => Dex.species.get(s.species).baseSpecies);
  return {sets, bytes, species};
}

const reserved = [...new Set([...Object.keys(base.hashes), ...earlier.teams, ...earlier.training_teams])];
const guard = reserved.map(file => sheet(file).species);
const own = [...base.own_train];
const hashes = {...base.hashes};
const order = fs.readdirSync(path.join(root, bank)).filter(f => /^MB\d+\.txt$/.test(f))
  .sort((a, b) => sha('mb-diversity-v1/' + a).localeCompare(sha('mb-diversity-v1/' + b)));
for (const file of order) {
  const full = bank + '/' + file;
  const {sets, bytes, species} = sheet(full);
  if (sets.length !== 6 || sets.some(s => s.moves.length !== 4 || !s.ability || !s.nature)) continue;
  if (guard.some(team => team.filter(mon => species.includes(mon)).length > 3)) continue;
  if (validator.validateTeam(sets)?.length) continue;
  own.push(full);
  hashes[full] = sha(bytes);
  guard.push(species);
  if (own.length === 16) break;
}
if (own.length !== 16) throw Error('Not enough disjoint legal tournament rosters');
const protocol = {
  source: base.source, simulator: base.simulator, format: base.format, own_train: own,
  foe_train: base.foe_train, own_test: base.own_test, foe_test: base.foe_test, hashes,
  steps: 262144, training_seeds: base.training_seeds, arms: ['shared', 'moe'],
  opponent: 'Frozen released BC with native learned preview on both sides throughout.',
  schedule: '16 independent roster streams. Every four updates, permute all 16 by training seed and block, collect four streams per update and 64 decisions per stream. Each roster gets exactly 16384 decisions. Keep original optimizer, GAE horizon, terminal reward and BC initialization.',
  selection: 'Keep original four. Hash-order the pinned tournament bank; require legal complete teams and at most three base species in common with each reserved roster and each earlier selection.',
  evaluation: 'Original forward and reverse held-out tournaments, same 16 games per pair, sampled actions, seats, simulation seeds and previews. Equal weight for each direction.',
  primary: 'Paired held-out normal minus cyclically rotated expert identities for 16-team training; compare that effect with the four-team control.',
  secondary: 'MoE minus shared, 16-team minus four-team training within architecture, and normal MoE minus its fine-tuned backbone with residuals disabled. Use original crossed 6000-draw paired bootstrap.'
};
const dest = path.join(root, 'results/sketch5-mb-team-diversity/protocol.json');
const text = JSON.stringify(protocol, null, 2) + '\n';
fs.mkdirSync(path.dirname(dest), {recursive:true});
if (fs.existsSync(dest) && fs.readFileSync(dest,'utf8') !== text) throw Error('Roster lock changed.');
fs.writeFileSync(dest, text);
const union = files => new Set(files.flatMap(file => sheet(file).species));
const oldSpecies = union(base.own_train);
const newSpecies = union(own);
const testSpecies = union([...base.own_test, ...base.foe_test]);
const countCovered = pool => [...testSpecies].filter(mon => pool.has(mon)).length;
const lines = [
  '# Tournament rosters', '',
  `The original four learner teams contain ${oldSpecies.size} distinct base species. The 16-team pool contains ${newSpecies.size}. Of the ${testSpecies.size} distinct base species on held-out teams, ${countCovered(oldSpecies)} appear on the original learner rosters and ${countCovered(newSpecies)} appear on the expanded learner rosters. These counts use the full six-slot sheets. Preview determines which four enter each game.`,
  '', '| Assignment | Team | Six Pokémon in sheet order |', '|---|---|---|',
];
for (const [role, group] of [['Original learner', base.own_train], ['Added learner', own.slice(4)],
    ['Training opponent', base.foe_train], ['Held-out group of four', base.own_test],
    ['Held-out group of eight', base.foe_test]]) {
  for (const file of group) {
    lines.push(`| ${role} | ${path.basename(file, '.txt')} | ${sheet(file).sets.map(s => s.species).join(', ')} |`);
  }
}
fs.writeFileSync(path.join(path.dirname(dest), 'rosters.md'), lines.join('\n') + '\n');
console.log(own.map(x => path.basename(x,'.txt')).join(', '));
