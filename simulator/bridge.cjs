// JSON-lines transport over the pinned Showdown simulator. No network server.
const path = require('node:path');
const readline = require('node:readline');
const root = path.resolve(process.argv[2]);
const {Battle, Teams, TeamValidator} = require(root + '/dist/sim');
const {extractChannelMessages} = require(root + '/dist/sim/battle');
let battle;
let requests;
let views;
let spectator;
let errors;

function receive(type, data) {
  const text = Array.isArray(data) ? data.join('\n') : data;
  if (type === 'update') {
    const channels = extractChannelMessages(text, [0, 1, 2]);
    spectator.push(...channels[0]);
    views.p1.push(...channels[1]);
    views.p2.push(...channels[2]);
  } else if (type === 'sideupdate') {
    const at = text.indexOf('\n');
    const side = text.slice(0, at);
    for (const line of text.slice(at + 1).split('\n')) {
      if (line.startsWith('|request|')) requests[side] = JSON.parse(line.slice(9));
      else if (line.startsWith('|error|')) errors.push(side + ': ' + line);
    }
  }
}

function handle(command) {
  views = {p1: [], p2: []};
  spectator = [];
  errors = [];
  if (command.op === 'reset') {
    if (battle) battle.destroy();
    const teams = command.teams.map(text => Teams.import(text));
    const validator = new TeamValidator(command.format);
    teams.forEach((team, index) => {
      const problems = validator.validateTeam(team);
      if (problems) throw new Error(`Team ${index + 1}: ${problems.join('; ')}`);
    });
    requests = {p1: null, p2: null};
    battle = new Battle({formatid: command.format, seed: command.seed, send: receive});
    battle.setPlayer('p1', {name: 'Learner', team: teams[0]});
    battle.setPlayer('p2', {name: 'Opponent', team: teams[1]});
  } else if (command.op === 'step') {
    if (!battle || battle.ended) throw new Error('No running battle');
    // Check the original simultaneous requests before committing either side's choice.
    for (const side of ['p1', 'p2']) {
      if (requests[side] && !requests[side].wait && !command.choices[side]) {
        throw new Error(`Missing choice for ${side}`);
      }
    }
    for (const [side, choice] of Object.entries(command.choices)) {
      requests[side] = null;
      if (!battle.choose(side, choice)) {
        // Hidden traps and disables reveal a revised private request to this player.
        // The other player's submitted choice stays committed to the same turn.
        const error = errors.at(-1);
        if (!requests[side]?.update || !error?.startsWith(side + ': |error|[Unavailable choice]')) {
          throw new Error(error || `Rejected ${side}: ${choice}`);
        }
        errors.pop();
      }
    }
  } else {
    throw new Error('Unknown operation');
  }
  battle.sendUpdates();
  if (errors.length) throw new Error(errors.join('; '));
  return {views, requests, spectator, turn: battle.turn, ended: battle.ended,
    winner: battle.ended ? battle.winner : null};
}

const input = readline.createInterface({input: process.stdin});
input.on('line', line => {
  try { process.stdout.write(JSON.stringify({ok: true, ...handle(JSON.parse(line))}) + '\n'); }
  catch (error) { process.stdout.write(JSON.stringify({ok: false, error: error.stack}) + '\n'); }
});
input.on('close', () => { if (battle) battle.destroy(); });
