// Independent hypothetical battles. Input contains only the learner's public-position projection.
const readline = require("node:readline");
const crypto = require("node:crypto");
const {Battle, Dex} = require(process.argv[2] + "/dist/sim");
const id = s => String(s || "").toLowerCase().replace(/[^a-z0-9]/g, "");
function words(key) {
  const bytes = crypto.createHash("sha256").update(key).digest();
  return [0, 2, 4, 6].map(i => bytes.readUInt16LE(i));
}
function uniform(key) { return (words(key)[0]+.5)/65536; }

function setFor(mon, key) {
  if (!mon) return {species:"Magikarp", ability:"No Ability", moves:["Splash"], level:50};
  const species = Dex.species.get(mon.species);
  // Use model-specific estimates for opposing private spreads. The learner provides its own stats.
  const physical = mon.moves.reduce((n,m) => {
    const move = Dex.moves.get(m);
    return n + (move.category === "Physical" ? 1 : -1 * (move.category === "Special")) * move.basePower;
  }, 0) > 0;
  const fast = species.baseStats.spe >= 80 && uniform(key) < .75;
  const evs = {hp:fast?2:32, atk:0, def:0, spa:0, spd:fast?0:2, spe:fast?32:0};
  evs[physical?"atk":"spa"] = 32;
  const nature = fast ? (physical?"Jolly":"Timid") : (physical?"Adamant":"Modest");
  return {name:mon.name || species.name, species:species.isMega ? species.baseSpecies : species.name, ability:mon.ability,
          item:mon.item, moves:mon.moves, evs, nature, level:50};
}

function patchMon(b, p, visible) {
  if (!visible) {
    p.hp=0; p.fainted=true; p.faintQueued=false;
    return;
  }
  const species=b.dex.species.get(visible.species);
  if (species.isMega && p.species.id!==species.id) {
    p.formeChange(species, null, true);
    for(const q of p.side.pokemon) q.canMegaEvo=null;
  }
  if (visible.stats && visible.stats.hp) {
    p.maxhp=visible.stats.hp; p.baseMaxhp=p.maxhp;
    for(const k of ["atk","def","spa","spd","spe"]) if (visible.stats[k]) {
      p.storedStats[k]=visible.stats[k]; p.baseStoredStats[k]=visible.stats[k];
    }
  }
  p.hp=Math.max(1,Math.round(visible.hp*p.maxhp));
  p.status=visible.status;
  p.statusState=b.initEffectState({id:p.status,target:p,time:2,startTime:2,stage:1});
  p.boosts={...p.boosts,...visible.boosts};
  p.item=id(visible.item); if (visible.ability) p.ability=id(visible.ability);
  p.activeTurns=visible.first?1:3; p.activeMoveActions=visible.first?0:2;
  if (visible.last) p.lastMove=b.dex.moves.get(visible.last);
  for(const slot of p.moveSlots) {
    const i=visible.moves.indexOf(slot.id);
    if(i>=0) slot.pp=Math.max(0,Math.min(slot.maxpp,visible.pp[i]));
  }
  if(visible.protect>0) {
    p.volatiles.stall=b.initEffectState({id:"stall",target:p,duration:2,
                                        counter:Math.min(729,3 ** (visible.protect+1))});
  }
}

function world(pos,key) {
  const b=new Battle({formatid:"gen9championsvgc2026regmb",seed:words(key+"init")});
  const boards=[pos.own.slice(),pos.foe.slice()];
  if(pos.bench){
    boards[0].push(...pos.bench[0]);boards[1].push(...pos.bench[1]);
    const guesses=pos.unknown.slice().sort((x,y)=>uniform(key+x.species)-uniform(key+y.species));
    boards[1].push(...guesses.slice(0,pos.vacancies));
  }
  boards.forEach((mons,j) => b.setPlayer("p"+(j+1),
    {name:j===0?"Us":"Them",team:mons.map((m,i)=>setFor(m,key+j+"-"+i))}));
  boards.forEach((mons,j)=>{
    if(!b.choose("p"+(j+1),"team "+mons.map((_,i)=>i+1).join("")))
      throw Error("Invalid hypothetical lead/bench order");
  });
  b.turn=pos.turn;
  boards.forEach((mons,j) => {
    const side=b.sides[j];
    mons.forEach((m,i)=>patchMon(b,side.pokemon[i],m));
    side.pokemonLeft=mons.filter(Boolean).length;
    side.sideConditions={};
    for(const [name,amount] of Object.entries(pos.sides[j])) {
      const cond=id(name);
      if(!["tailwind","reflect","lightscreen","auroraveil","safeguard","mist"].includes(cond)) continue;
      const duration=(cond==="tailwind"?4:5)-Math.max(0,pos.turn-amount);
      side.sideConditions[cond]=b.initEffectState({id:cond,target:side,duration:Math.max(1,duration)});
    }
  });
  const weather=Object.entries(pos.weather)[0];
  b.field.weather=weather?id(weather[0]):"";
  b.field.weatherState=b.initEffectState({id:b.field.weather,duration:weather?Math.max(1,5-pos.turn+weather[1]):0});
  b.field.terrain=""; b.field.terrainState=b.initEffectState({id:""}); b.field.pseudoWeather={};
  for(const [name,start] of Object.entries(pos.field)){
    const effect=id(name); const state=b.initEffectState({id:effect,duration:Math.max(1,5-pos.turn+start)});
    if(effect.endsWith("terrain")){b.field.terrain=effect;b.field.terrainState=state;}
    else if(["trickroom","magicroom","wonderroom","gravity"].includes(effect)) b.field.pseudoWeather[effect]=state;
  }
  b.makeRequest("move");
  return b;
}

function pressure(b,p,t,move) {
  if(!t || t.fainted) return 0;
  if(!move.basePower || !b.dex.getImmunity(move.type,t)) return 0;
  const physical=move.category==="Physical";
  const ratio=p.getStat(physical?"atk":"spa")/Math.max(1,t.getStat(physical?"def":"spd"));
  const mult=2**b.dex.getEffectiveness(move.type,t) * (p.hasType(move.type)?1.5:1);
  const frac=(.44*move.basePower*ratio+2)*mult/t.maxhp;
  const hp=t.hp/t.maxhp;
  return Math.min(hp,frac)+.35*(frac>=hp)+.07*Math.max(0,move.priority);
}
function options(b,side,i){
  const p=side.active[i];
  if(!p || p.fainted) return [{cmd:"pass",score:0}];
  const req=side.activeRequest.active[i];
  const result=[];
  for(const slot of req.moves){
    if(slot.disabled || slot.pp===0) continue;
    const m=b.dex.moves.get(slot.id), tgt=slot.target||m.target;
    let targets=[0];
    if(["normal","any","adjacentFoe"].includes(tgt))
      targets=side.foe.active.flatMap((q,j)=>q&&!q.fainted?[j+1]:[]);
    if(tgt==="adjacentAlly" || tgt==="adjacentAllyOrSelf")
      targets=side.active.flatMap((q,j)=>q&&!q.fainted&&(j!==i||tgt==="adjacentAllyOrSelf")?[-j-1]:[]);
    for(const t of targets) {
      let score=0;
      if(t>0) score=pressure(b,p,side.foe.active[t-1],m);
      else if(t===0) score=side.foe.active.reduce((v,q)=>v+.75*pressure(b,p,q,m),0);
      if(["protect","detect","spikyshield"].includes(m.id)) score=p.volatiles.stall?-.3:.23;
      if(m.id==="fakeout") score+=p.activeMoveActions===0?.65:-1;
      if(!m.basePower && ["spore","sleeppowder","willowisp","thunderwave"].includes(m.id) && t>0 && !side.foe.active[t-1].status) score+=.43;
      if(m.heal) score+=.45*(1-p.hp/p.maxhp);
      if(m.boosts && m.target==="self") score+=.25*p.hp/p.maxhp;
      result.push({cmd:"move "+m.id+(t?" "+t:""),score});
      if(req.canMegaEvo) result.push({cmd:"move "+m.id+" mega"+(t?" "+t:""),score:score+.22,mega:true});
    }
  }
  if(!req.trapped){
    for(const reserve of side.pokemon.slice(2)) if(reserve.hp && !reserve.fainted){
      const threats=side.foe.active.filter(q=>q&&!q.fainted);
      const safer=threats.reduce((total,foe)=>{
        const attackOn=target=>Math.max(0,...foe.moveSlots.map(m=>pressure(b,foe,target,b.dex.moves.get(m.id))));
        return total+attackOn(p)-attackOn(reserve);
      },0);
      result.push({cmd:"switch "+reserve.name,score:-.35+.45*safer});
    }
  }
  return result;
}
function enemy(b,key) {
  const [a,c]=[options(b,b.p2,0),options(b,b.p2,1)];
  const choices=[];
  for(const x of a) for(const y of c) if(!(x.mega&&y.mega)){
    const cmd=x.cmd+","+y.cmd;
    const noise=-Math.log(-Math.log(uniform(key+cmd)));
    choices.push([x.score+y.score+.20*noise,cmd]);
  }
  choices.sort((x,y)=>y[0]-x[0]);
  for(const [,cmd] of choices) if(b.choose("p2",cmd)) return true;
  return false;
}
function snapshot(b) {
  function describe(p){return !p||p.fainted||!p.hp?null:{
    hp:p.hp/p.maxhp,status:p.status,boosts:p.boosts,baseSpeed:p.species.baseStats.spe,
    species:p.species.id,ability:p.ability,moves:p.moveSlots.map(x=>x.id)};}
  return {own:b.p1.active.map(describe),foe:b.p2.active.map(describe),
          bench:b.sides.map(s=>s.pokemon.filter(p=>!s.active.includes(p)).map(describe).filter(Boolean)),
          unknown:[],vacancies:0,
          field:b.field.pseudoWeather.trickroom?{trickroom:1}:{},
          sides:b.sides.map(s=>Object.fromEntries(Object.keys(s.sideConditions).map(k=>[k,1])))};
}
function forecast(input){
  const worlds=[];
  for(let rep=0;rep<4;rep++){
    const key=input.key+"-"+rep;
    const b=world(input.position,key);
    if(!enemy(b,key)){b.destroy();continue;}
    const serial=JSON.stringify(b.toJSON());
    const leaves=[];
    for(const order of input.orders){
      const c=Battle.fromJSON(serial);
      c.resetRNG(words(key+"turn"));
      if(!c.choose("p1",order)){c.destroy();break;}
      leaves.push(c.ended ? (c.winner==="Us"?1:c.winner==="Them"?0:.5) : snapshot(c));
      c.destroy();
    }
    b.destroy();
    if(leaves.length===input.orders.length)worlds.push(leaves);
  }
  return worlds;
}
const input=readline.createInterface({input:process.stdin});
input.on("line",line=>{
  try{process.stdout.write(JSON.stringify({worlds:forecast(JSON.parse(line))})+"\n");}
  catch(e){process.stdout.write(JSON.stringify({error:e.stack})+"\n");}
});
