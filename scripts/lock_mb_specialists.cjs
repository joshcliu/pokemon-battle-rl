// Assign a specialist using only public roster sheets before evaluation.
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const pools = require('../results/sketch5-mb-reward-factorial/protocol.json');
const {Teams, Dex} = require(path.join(root, '.cache/vgcbench-external/showdown/dist/sim'));

function features(file) {
  const sets = Teams.import(fs.readFileSync(path.join(root, file), 'utf8'));
  if (sets.length !== 6) throw Error('Expected six Pokémon in ' + file);
  return {
    name: path.basename(file, '.txt'),
    species: new Set(sets.map(s => Dex.species.get(s.species).baseSpecies)),
    moves: new Set(sets.flatMap(s => s.moves.map(m => Dex.moves.get(m).id))),
  };
}
function sharedCount(a, b) {
  return Array.from(a).filter(value => b.has(value)).length;
}
const train = pools.own_train.map(features);
const rows = [];
for (const file of [...pools.own_train, ...pools.own_test, ...pools.foe_test]) {
  const team = features(file);
  const candidates = train.map((other, i) => ({
    index: i, training_team: other.name,
    shared_species: sharedCount(team.species, other.species),
    shared_moves: sharedCount(team.moves, other.moves),
  }));
  candidates.sort((a,b) => b.shared_species-a.shared_species || b.shared_moves-a.shared_moves || a.index-b.index);
  rows.push({team: team.name, ...candidates[0], candidates});
}
const dest = path.join(root, 'results/sketch5-mb-roster-ppo/specialists.json');
const contents = JSON.stringify(rows, null, 2) + '\n';
if (fs.existsSync(dest) && fs.readFileSync(dest,'utf8') !== contents) throw Error('Existing specialist roster assignments differ.');
fs.writeFileSync(dest, contents);
