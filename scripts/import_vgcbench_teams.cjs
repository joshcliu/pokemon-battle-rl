// Freeze a reproducible holdout before examining any game outcomes.
const {createHash} = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const {Teams, TeamValidator, Dex} = require(path.join(root, '.cache/pokemon-showdown/dist/sim'));
const source = process.argv[2];
const dest = path.join(root, 'teams/vgcbench-mb-transfer');
const sha256 = value => createHash('sha256').update(value).digest('hex');

function rosterKeys(team) {
  const species = team.map(set => Dex.species.get(set.species).baseSpecies);
  return [species.toSorted().join('|'), species.slice(0, 4).toSorted().join('|')];
}

const seenSix = new Set();
const seenFour = new Set();
function reserve([six, four]) {
  seenSix.add(six);
  seenFour.add(four);
}

for (const dir of ['reg-mc-regionals', 'reg-mc-preview-expanded', '.']) {
  const base = path.join(root, 'teams', dir);
  for (const file of fs.readdirSync(base).filter(file => file.endsWith('.txt'))) {
    reserve(rosterKeys(Teams.import(fs.readFileSync(path.join(base, file), 'utf8'))));
  }
}

const bank = path.join(source, 'teams/reg_mb');
const files = fs.readdirSync(bank).filter(file => /^MB\d+\.txt$/.test(file));
const priority = file => sha256('mc461-external-v1/' + file);
files.sort((a, b) => priority(a).localeCompare(priority(b)));
const validators = ['mb', 'mc'].map(reg => new TeamValidator('gen9championsvgc2026reg' + reg));
const rows = [];
fs.mkdirSync(dest, {recursive: true});
for (const file of files) {
  const bytes = fs.readFileSync(path.join(bank, file));
  const text = bytes.toString('utf8');
  const team = Teams.import(text);
  const complete = set => set.moves.length === 4 && set.ability && set.nature &&
    Object.values(set.evs ?? {}).some(value => value > 0);
  if (team.length !== 6 || !team.every(complete)) continue;
  const keys = rosterKeys(team);
  if (seenSix.has(keys[0]) || seenFour.has(keys[1])) continue;
  if (validators.some(validator => validator.validateTeam(Teams.import(text))?.length)) continue;
  reserve(keys);
  fs.writeFileSync(path.join(dest, file), bytes);
  const gitBlob = createHash('sha1').update('blob ' + bytes.length + '\0').update(bytes).digest('hex');
  rows.push({
    team: file, source: 'teams/reg_mb/' + file, sha256: sha256(bytes), git_blob: gitBlob,
    base_species_six: keys[0], playable_four: keys[1],
  });
  if (rows.length === 12) break;
}
if (rows.length !== 12) throw Error('Insufficient legal unseen teams');
fs.copyFileSync(path.join(source, 'LICENSE'), path.join(dest, 'LICENSE'));
fs.writeFileSync(path.join(dest, 'source.json'), JSON.stringify({
  revision: 'd79f9532947ac114dce1dda2456a590afcd375b2', teams: rows,
}, null, 2) + '\n');
console.log('External team holdout:', rows.map(row => row.team).join(', '));
