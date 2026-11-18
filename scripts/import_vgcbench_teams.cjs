// Freeze an outcome-independent holdout from the authors' M-B team bank.
const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');
const root = path.resolve(__dirname, '..');
const {Teams, TeamValidator, Dex} = require(root + '/.cache/pokemon-showdown/dist/sim');
const src = process.argv[2];
const dest = path.join(root,'teams/vgcbench-mb-transfer');
const sha = text => createHash('sha256').update(text).digest('hex');
const gitBlob = bytes => createHash('sha1').update('blob '+bytes.length+'\0').update(bytes).digest('hex');
const keys = t => {
  const species = t.map(s => Dex.species.get(s.species).baseSpecies);
  return [species.toSorted().join('|'), species.slice(0,4).toSorted().join('|')];
};
const seen = [new Set(),new Set()];
function reserve(team) { keys(team).forEach((k,i) => seen[i].add(k)); }
for (const dir of ['reg-mc-regionals','reg-mc-preview-expanded','.']) {
  const base = path.join(root,'teams',dir);
  for (const file of fs.readdirSync(base).filter(f=>f.endsWith('.txt'))) {
    reserve(Teams.import(fs.readFileSync(path.join(base,file),'utf8')));
  }
}
const dir = path.join(src,'teams/reg_mb');
const files = fs.readdirSync(dir).filter(f=>/^MB\d+\.txt$/.test(f));
files.sort((a,b) => sha('mc461-external-v1/'+a).localeCompare(sha('mc461-external-v1/'+b)));
const validators = ['mb','mc'].map(r=>new TeamValidator('gen9championsvgc2026reg'+r));
const rows = [];
fs.mkdirSync(dest,{recursive:true});
for (const file of files) {
  const bytes = fs.readFileSync(path.join(dir,file));
  const text = bytes.toString('utf8');
  const team = Teams.import(text);
  if (team.length!==6 || team.some(s=>s.moves.length!==4 || !s.ability || !s.nature ||
      !Object.values(s.evs??{}).some(v=>v>0))) continue;
  const ks = keys(team);
  if (ks.some((k,i)=>seen[i].has(k))) continue;
  if (validators.some(v=>v.validateTeam(Teams.import(text))?.length)) continue;
  reserve(team);
  fs.writeFileSync(path.join(dest,file),bytes);
  rows.push({team:file,source:'teams/reg_mb/'+file,sha256:sha(bytes),git_blob:gitBlob(bytes),
             base_species_six:ks[0],playable_four:ks[1]});
  if(rows.length===12)break;
}
if (rows.length!==12) throw Error('Insufficient legal unseen teams');
fs.copyFileSync(path.join(src,'LICENSE'),path.join(dest,'LICENSE'));
fs.writeFileSync(path.join(dest,'source.json'),JSON.stringify({revision:'d79f9532947ac114dce1dda2456a590afcd375b2',teams:rows},null,2)+'\n');
console.log('External team holdout:',rows.map(r=>r.team).join(', '));
