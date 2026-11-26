/**
 * 0AD-Eyes replay camera (CAP-2). Loaded with the session page (session.xml loads every
 * script in gui/session/, alphabetically); the zz_ prefix makes it load after session.js,
 * whose register*Handler functions it calls at load time.
 *
 * Off unless the config key replaycam.enabled is "true"; record_replays.py passes it and
 * the rest with -conf=replaycam.<key>:<value>:
 *   player      the player whose perspective is shown (default 1): their fog, HUD and panels
 *   segments    comma-separated sim seconds where a recording segment starts (default "120")
 *   length      seconds of sim time per segment (default 240)
 *   shot        seconds the camera stays on one target (default 6)
 *   ffrate      sim rate while skipping to the next segment (default 10)
 *
 * It prints one line per state change to stdout, which the controller reads:
 *   REPLAYCAM SEGMENT_START <index> <simSeconds>
 *   REPLAYCAM SEGMENT_END <index> <simSeconds>
 *   REPLAYCAM DONE <simSeconds>
 *
 * The director mirrors the fork's CDatasetExporter::PickDirectorShot: entities in combat
 * weigh 4, foundations 2, workers 2, anything else 1; a category is drawn by weight and an
 * entity uniformly inside it. The target is selected so the HUD shows its panel.
 */
var g_ReplayCam = {
	"config": function(key, fallback)
	{
		const value = Engine.ConfigDB_GetValue("user", "replaycam." + key);
		return value === "" || value === undefined ? fallback : value;
	},
	"started": false,
	"done": false,
	"segment": 0,
	"recording": false,
	"segmentEnd": 0,
	"nextShot": 0,
	"maxSample": 250
};

g_ReplayCam.enabled = g_ReplayCam.config("enabled", "false") == "true";
g_ReplayCam.player = +g_ReplayCam.config("player", "1");
g_ReplayCam.segments = g_ReplayCam.config("segments", "120").split(",").map(s => +s).filter(s => s >= 0).sort((a, b) => a - b);
g_ReplayCam.length = +g_ReplayCam.config("length", "240");
g_ReplayCam.shot = +g_ReplayCam.config("shot", "6");
g_ReplayCam.ffrate = +g_ReplayCam.config("ffrate", "10");

g_ReplayCam.log = function(message)
{
	print("REPLAYCAM " + message + "\n");
};

g_ReplayCam.start = function()
{
	this.started = true;
	// Take the player's perspective so the HUD (resources, command panel) looks like play.
	g_PlayerViewControl.setChangePerspective(true);
	g_PlayerViewControl.selectViewPlayer(this.player + 1);
	this.log("START player=" + this.player + " segments=" + this.segments.join(","));
};

g_ReplayCam.category = function(state)
{
	if (state.foundation)
		return 1;
	const unitState = state.unitAI && state.unitAI.state || "";
	if (unitState.indexOf("COMBAT") != -1)
		return 0;
	if (/GATHER|REPAIR|RETURNRESOURCE|TRADE|GARRISON|HEAL/.test(unitState))
		return 2;
	return 3;
};

g_ReplayCam.pickShot = function()
{
	let ents = Engine.GuiInterfaceCall("GetPlayerEntities");
	if (!ents || !ents.length)
		return undefined;
	// Shuffle-sample so big late-game armies stay cheap to inspect.
	if (ents.length > this.maxSample)
	{
		ents = ents.slice();
		for (let i = 0; i < this.maxSample; ++i)
		{
			const j = i + Math.floor(Math.random() * (ents.length - i));
			[ents[i], ents[j]] = [ents[j], ents[i]];
		}
		ents.length = this.maxSample;
	}
	const weights = [4, 2, 2, 1];
	const buckets = [[], [], [], []];
	for (const item of Engine.GuiInterfaceCall("GetMultipleEntityStates", ents))
	{
		const state = item.state;
		// No position = garrisoned or otherwise out of the world; no identity = nothing to select.
		if (!state || !state.position || !state.identity || state.player != g_ViewedPlayer)
			continue;
		buckets[this.category(state)].push(state);
	}
	let total = 0;
	for (let c = 0; c < 4; ++c)
		if (buckets[c].length)
			total += weights[c];
	if (!total)
		return undefined;
	let roll = Math.random() * total;
	for (let c = 0; c < 4; ++c)
	{
		if (!buckets[c].length)
			continue;
		roll -= weights[c];
		if (roll <= 0)
			return buckets[c][Math.floor(Math.random() * buckets[c].length)];
	}
	return undefined;
};

g_ReplayCam.nextShotNow = function()
{
	const state = this.pickShot();
	if (!state)
		return;
	g_Selection.reset();
	g_Selection.addList([state.id]);
	Engine.CameraMoveTo(state.position.x, state.position.z);
	// Units move, so keep them in view; buildings stay put.
	Engine.CameraFollow(state.unitAI ? state.id : 0);
};

g_ReplayCam.update = function()
{
	if (!this.enabled || this.done)
		return;
	const simState = GetSimState();
	if (!simState)
		return;
	if (!this.started)
		this.start();

	const now = simState.timeElapsed / 1000;

	if (this.recording)
	{
		if (now >= this.segmentEnd)
		{
			this.recording = false;
			this.log("SEGMENT_END " + this.segment + " " + now.toFixed(1));
			++this.segment;
		}
		else if (now >= this.nextShot)
		{
			this.nextShot = now + this.shot;
			this.nextShotNow();
		}
		if (this.recording)
			return;
	}

	if (this.segment >= this.segments.length)
	{
		this.done = true;
		Engine.SetSimRate(1);
		this.log("DONE " + now.toFixed(1));
		return;
	}

	// Skip ahead to the next segment, slowing down just before it so it starts at 1x.
	const start = this.segments[this.segment];
	if (now < start - 3)
	{
		if (Engine.GetSimRate() != this.ffrate)
			Engine.SetSimRate(this.ffrate);
		return;
	}
	if (now < start)
	{
		if (Engine.GetSimRate() != 1)
			Engine.SetSimRate(1);
		return;
	}

	Engine.SetSimRate(1);
	this.recording = true;
	this.segmentEnd = now + this.length;
	this.nextShot = now + this.shot;
	this.nextShotNow();
	this.log("SEGMENT_START " + this.segment + " " + now.toFixed(1));
};

// The replay ending (or the viewed player losing) ends the current segment cleanly.
g_ReplayCam.finish = function()
{
	if (!this.enabled || this.done)
		return;
	const simState = GetSimState();
	const now = simState ? simState.timeElapsed / 1000 : 0;
	if (this.recording)
		this.log("SEGMENT_END " + this.segment + " " + now.toFixed(1));
	this.recording = false;
	this.done = true;
	this.log("DONE " + now.toFixed(1));
};

registerSimulationUpdateHandler(() => g_ReplayCam.update());
registerPlayersFinishedHandler(playerIDs => {
	if (playerIDs.indexOf(g_ReplayCam.player) != -1)
		g_ReplayCam.finish();
});
