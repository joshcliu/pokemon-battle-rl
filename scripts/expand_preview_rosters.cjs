// Import additional top-64 regional rosters from the same frozen VGCPastes release.
const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {gunzipSync} = require('node:zlib');
const root = path.resolve(__dirname, '..');
const {Teams, TeamValidator, Dex} = require(root + '/.cache/pokemon-showdown/dist/sim');
const out = path.join(root, 'teams/reg-mc-preview-expanded');
const [metaPath, pastesPath] = process.argv.slice(2);
function loadArchive(file, expected) {
  const bytes = fs.readFileSync(file);
  const actual = createHash('sha1').update('blob ' + bytes.length + '\0').update(bytes).digest('hex');
  if (actual !== expected) throw Error('Unexpected source archive: ' + file);
  return JSON.parse(gunzipSync(bytes));
}
const meta = loadArchive(metaPath, 'c66558ee95427c4d8ddaa9cd38c85a1ef9f1a434');
const pastes = loadArchive(pastesPath, '66d555eadfc9a4851dfe65ae12ab6d53e9228b09');
const validator = new TeamValidator('gen9championsvgc2026regmc');
const sixes = new Set(), fours = new Set();
function keys(team) {
  const names = team.map(s => Dex.species.get(s.species).baseSpecies);
  return [names.toSorted().join('|'), names.slice(0, 4).toSorted().join('|')];
}
const original = path.join(root,'teams/reg-mc-regionals');
for (const file of [...fs.readdirSync(original).filter(f=>f.endsWith('.txt')).map(f=>path.join(original,f)),
                    ...['sun','balance'].map(f=>path.join(root,'teams',f+'.txt'))]) {
  const [six, four] = keys(Teams.import(fs.readFileSync(file,'utf8')));
  sixes.add(six); fours.add(four);
}
function rank(label) {
  if (label === 'Champion') return 1;
  if (label === 'Runner Up') return 2;
  return Number(/^(?:Top )?(\d+)(?:st|nd|rd|th)?$/.exec(label)?.[1] ?? Infinity);
}
const candidates = meta.filter(r=> /Regional 2027$/i.test(r.event) && r.has_evs &&
  !/senior|junior/i.test(r.description) && rank(r.rank) <= 64);
candidates.sort((a,b)=>rank(a.rank)-rank(b.rank) || a.team_id.localeCompare(b.team_id, 'en'));
const rows = [];
fs.mkdirSync(out,{recursive:true});
for (const r of candidates) {
  const raw = pastes[r.pokepaste.replace(/\/$/,'').split('/').at(-1)];
  if (!raw) continue;
  const text = raw.split(/\r?\n/).map(s=>s.trimEnd()).join('\n').trim()+'\n';
  const team = Teams.import(text);
  if (team.length !== 6 || team.some(s=>!s.ability || !s.nature || s.moves.length !== 4 ||
      !Object.values(s.evs || {}).some(v=>v>0)) || validator.validateTeam(team)?.length) continue;
  const [six,four] = keys(team);
  if (sixes.has(six) || fours.has(four)) continue;
  sixes.add(six); fours.add(four);
  fs.writeFileSync(path.join(out,r.team_id+'.txt'),text);
  const {team_id,date,event,rank:placement,player,pokepaste} = r;
  rows.push({team_id,date,event,placement,player,pokepaste,base_species_six:six,playable_four:four});
}
fs.writeFileSync(path.join(out,'source.json'),JSON.stringify(rows,null,2)+'\n');
console.log('Eligible records:',candidates.length,'new validated unique rosters:',rows.length);
