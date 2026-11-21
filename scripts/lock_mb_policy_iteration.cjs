// No outcomes enter this split or team novelty check.
const fs = require("node:fs");
const path = require("node:path");
const {createHash} = require("node:crypto");
const root = path.resolve(__dirname, "..");
const prior = require("../results/sketch5-mb-team-diversity/protocol.json");
const dev = require("../results/sketch5-vgcbench-bc/protocol.json");
const simulator = path.join(root, ".cache/vgcbench-external/showdown/dist/sim");
const {Teams, TeamValidator, Dex} = require(simulator);
const validator = new TeamValidator(prior.format);
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const bank = ".cache/vgcbench-external/source/teams/reg_mb";
const custom = "teams/reg-mb-original/sun-room.txt";
const read = p => fs.readFileSync(path.join(root,p),"utf8");
function mons(file) {
  return Teams.import(read(file)).map(s => Dex.species.get(s.species).baseSpecies);
}
function overlap(a,b) { return a.filter(x => b.includes(x)).length; }
function legal(file) {
  const sets=Teams.import(read(file));
  return sets.length===6 && sets.every(s=>s.moves.length===4 && s.ability && s.nature)
    && !validator.validateTeam(sets)?.length;
}
const errors=validator.validateTeam(Teams.import(read(custom)));
if (errors?.length) throw Error(errors.join("\n"));
const oldFiles=[...new Set([...Object.keys(prior.hashes),...dev.teams,...dev.training_teams])];
const bankFiles=fs.readdirSync(path.join(root,bank)).filter(x=>/^MB\d+\.txt$/.test(x)).sort()
    .map(n=>path.join(bank,n));
const old=oldFiles.map(mons);
const team=mons(custom);
const candidates=bankFiles.map(file=>({file,mons:mons(file)}));
const maximum=(pool,x)=>Math.max(...pool.map(t=>overlap(t,x)));
const novelty={old_study_maximum_overlap:maximum(old,team),bank_maximum_overlap:maximum(candidates.map(c=>c.mons),team),bank_rosters:candidates.length};
if(novelty.bank_maximum_overlap>=6||novelty.old_study_maximum_overlap>3)throw Error("Not sufficiently distinct: "+JSON.stringify(novelty));
const forbidden=[...old,team];
function partition(label,n) {
  const pool=[...candidates].sort((a,b)=>hash(label+a.file).localeCompare(hash(label+b.file)));
  const picked=[];
  for(const c of pool) {
    if(forbidden.some(x=>overlap(x,c.mons)>3) || picked.some(x=>overlap(x.mons,c.mons)>4) || !legal(c.file))continue;
    picked.push(c);
    if(picked.length===n)break;
  }
  if(picked.length!==n)throw Error("Insufficient eligible teams for "+label+": "+picked.length);
  forbidden.push(...picked.map(x=>x.mons));
  return picked.map(c=>c.file);
}
const confirm=partition("policy-iteration-confirm-v1",16);
const validation=partition("policy-iteration-validation-v1",12);
const expand=partition("policy-iteration-train-v1",24);
const spec={format:prior.format,source:prior.source,simulator:prior.simulator,own_train:[...prior.own_train,...expand.slice(0,12)],
  foe_train:[...prior.foe_train,...expand.slice(12)],validation_a:validation.slice(0,4),validation_b:validation.slice(4),
  confirmation_a:confirm.slice(0,8),confirmation_b:confirm.slice(8),custom,novelty,
  training_seeds:[2,3,4,5],hashes:Object.fromEntries([...new Set([...oldFiles,...confirm,...validation,...expand,custom])].map(p=>[p,hash(fs.readFileSync(path.join(root,p)))])),
  primary:"Frozen sampled native BC opponent; standard native BC preview unless preview is an explicitly trained component. Same open-sheet legal information. Reciprocal role-weighted win fraction incl draws as non-wins. Report Wilson or crossed clustered intervals and per-team results.",
  selection:"Tune only on training and development. Advance a fixed configuration on four-seed validation before confirmation; promote at >=80% reciprocal development. Run original reciprocal 4x8 schedule for direct comparison and a new 8x8 reciprocal pool with independent seeds. Original Sun Room gets a separate zero-shot evaluation against fresh teams. Never substitute a single strong roster or one direction for the general win rate.",
  transfer:"Report separate generalization to an independently trained battle opponent. Never use confirmation outcomes for candidate selection; retire a confirmation pool before another search round if tested.",
};
const out=path.join(root,"results/sketch5-mb-policy-iteration/protocol.json");
fs.mkdirSync(path.dirname(out),{recursive:true});
const bytes=JSON.stringify(spec,null,2)+"\n";
if(fs.existsSync(out)&&fs.readFileSync(out,"utf8")!==bytes)throw Error("Existing split differs");
fs.writeFileSync(out,bytes);
console.log("Legal custom team:",team.join(", "),JSON.stringify(novelty));
console.log("Train",spec.own_train.length,spec.foe_train.length,"development",validation.length,"confirmation reserved",confirm.length);
