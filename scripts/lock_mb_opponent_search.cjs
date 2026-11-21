// Select the second prospective test pool without opening any first-round game records.
const fs = require("node:fs");
const path = require("node:path");
const {createHash} = require("node:crypto");
const root = path.resolve(__dirname, "..");
const first = require("../results/sketch5-mb-policy-iteration/protocol.json");
const {Teams, TeamValidator, Dex} = require(path.join(root, ".cache/vgcbench-external/showdown/dist/sim"));
const validator = new TeamValidator(first.format);
const hash = x => createHash("sha256").update(x).digest("hex");
const bank = ".cache/vgcbench-external/source/teams/reg_mb";
const bytes = p => fs.readFileSync(path.join(root,p));
const sets = p => Teams.import(bytes(p).toString());
const species = t => t.map(s => Dex.species.get(s.species).baseSpecies);
const used = Object.keys(first.hashes).map(p => species(sets(p)));
const names = fs.readdirSync(path.join(root,bank)).filter(n => /^MB\d+\.txt$/.test(n))
  .map(n=>path.join(bank,n)).sort((a,b)=>hash("opponent-search-v2:"+a).localeCompare(hash("opponent-search-v2:"+b)));
const picked = [];
const overlap = (a,b) => a.filter(x => b.includes(x)).length;
for (const file of names) {
  const roster=sets(file), mons=species(roster);
  if (used.some(x=>overlap(x,mons)>3) || picked.some(x=>overlap(x.mons,mons)>4)) continue;
  if(roster.length!==6 || !roster.every(s=>s.moves.length===4 && s.ability && s.nature) || validator.validateTeam(roster)?.length) continue;
  picked.push({file,mons});
  if(picked.length===16) break;
}
if(picked.length!==16) throw Error("Only "+picked.length+" eligible fresh rosters");
const files=picked.map(x=>x.file);
const spec={
  previous_protocol_sha256:hash(bytes("results/sketch5-mb-policy-iteration/protocol.json")),
  first_confirmation_retired:true, format:first.format, training_seeds:first.training_seeds,
  own_train:first.own_train,foe_train:first.foe_train,development_a:first.validation_a,development_b:first.validation_b,
  confirmation_a:files.slice(0,8), confirmation_b:files.slice(8), custom:first.custom,
  fresh_roster_sha256:Object.fromEntries(files.map(f=>[f,hash(bytes(f))])),
  max_species_overlap_with_earlier_rosters:Math.max(...picked.flatMap(x=>used.map(t=>overlap(t,x.mons)))),
  max_species_overlap_within_round:Math.max(...picked.flatMap((x,i)=>picked.slice(i+1).map(y=>overlap(x.mons,y.mons)))),
  selection:"Select on the original development 4-by-8 only. Require at least 82% over all 2,048 paired games, above the first candidate's 80.71%. Reuse identical four pilot seeds, preview and simulator seeds for development. Freeze the selected implementation and weights before scoring the reserved 8-by-8 reciprocal test. Report both rounds, including a miss.",
  test:"Score all 128 ordered pairs, 16 games per training seed, four seeds, balanced seats. Score the original custom roster separately against all 16. Compare with the original sampled battle PPO on each schedule, and evaluate against the independent PPO opponent. Report crossed roster, seed and game intervals.",
  simulation_bases:{primary:670000000,custom:680000000,transfer:690000000}
};
const out=path.join(root,"results/sketch5-mb-opponent-search/protocol.json");
fs.mkdirSync(path.dirname(out),{recursive:true});
const data=JSON.stringify(spec,null,2)+"\n";
if(fs.existsSync(out)){if(fs.readFileSync(out,"utf8")!==data)throw Error("Different protocol already locked");}
else fs.writeFileSync(out,data,{flag:"wx"});
console.log(files.map(x=>path.basename(x,".txt")).join(" "),spec.max_species_overlap_with_earlier_rosters);
