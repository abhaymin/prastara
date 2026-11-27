# Clusters: several machines as one Prastara

Owner's request (2026-11-10): machines on the same network, or remote ones the user pairs with keys (peer to peer),
work together. A model's layers are shared across them, for quicker requests and longer context, as exo and other
cluster tools do, old phones included.

**State: phase 1 (request routing), part 1 (nodes and joining) and encryption with the paired keys built and tested
on one machine (2026-11-11); measurements on a second machine are next.** This page is the design and the
order of the work. Each step is measured on real machines before the next starts.

### Part 1 as built: nodes and joining (`crates/am2strata/src/cluster.rs`)

- **A worker:** `"cluster": {"role": "worker"}` in its run config (and optionally `"name"`).
  - It has an ed25519 identity in `<data>/node-key` (0600, made on first start). Its node id is the first 16 bytes
    of the SHA-256 of the public key.
  - It announces itself on the LAN by mDNS as `_am2strata._tcp` (TXT: name, version, node_id), using the
    `mdns-sd` crate (Apache-2.0 OR MIT).
  - It shows a 6-digit pairing code that changes every 10 minutes and after each use. The code appears in its log
    and its own Settings › Cluster, and only to its own main key.
- **The master:**
  - It browses for `_am2strata._tcp`. `GET /api/cluster` lists the servers found and not paired as "found";
    Settings › Cluster shows them as "wants to join", with a code field and a Pair button. A worker reached through
    a VPN can be paired by its address.
  - The address used is a private LAN address first, then a VPN's, then IPv6.
- **Pairing:**
  1. The master's `POST /api/cluster/pair {"url" or "node_id", "code"}` (main key, own page) sends the code to the
     worker's `POST /api/cluster/join`.
  2. The worker compares the code in constant time. Five wrong codes lock pairing for 10 minutes, even for the
     right code.
  3. For the right code it mints a random key for that master alone, keeps only the key's SHA-256 in
     `<data>/cluster-masters.json` (0600), and answers the key once.
  4. The master keeps the worker and its key in `<data>/cluster-workers.json` (0600), merged with any run-config
     workers at start, and polls it at once (no restart).
- **What a master's key opens on the worker:** the model APIs, plus the model list and queue the master polls.
  Never the worker's settings or its own pairing code. The worker's main key never leaves it.
- **Removing:** `DELETE /api/cluster/workers/{name}` on the master stops using the worker, and tells it to forget
  that master (`POST /api/cluster/forget` with the master's key). After that the key is refused.
- `AM2STRATA_NO_MDNS=1` turns mDNS off (the tests set it).
- **Tested:**
  - `a_worker_pairs_by_its_code_and_is_removed_cleanly`, on two servers on one machine: a wrong code is refused;
    the right code pairs; the key files are 0600; the worker keeps only the hash; the minted key opens
    `/v1/models`, `/api/models` and `/api/queue` but not `/api/cluster` or `/api/system`; a chat for the worker's
    model is answered by it through the master; removal makes the key refused; five wrong codes lock pairing.
  - Unit tests: the node identity, the code and the lock, the address preference.
  - `security_every_route_needs_the_key` names `/api/cluster/join` as the one route proven by a code (refused on a
    server that is not a worker).
- **Checked by hand on this PC (2026-11-11):** a worker and a master as two processes, mDNS on. The master found the
  worker (name, node id, version) with no configuration. It took the worker's Tailscale address, the first IPv4 the
  worker announced, which is why private LAN addresses now go first.
- **Security:** pairing needs physical or remote access to the worker's code. Each master has its own revocable
  key. Traffic is still plain HTTP with that key, so across a network you don't trust, use your own VPN.
  **Encryption with the paired keys (Noise or TLS pinned to the node ids) is the next step.**

### Phase 1 as built (`crates/am2strata/src/cluster.rs`)

- **Configuration:** the master's run config has
  `"cluster": {"workers": {"pc2": {"url": "http://host:8000", "key": "env:PC2_KEY"}}}`. A worker is an ordinary
  Prastara with its own API key, which only the master is given.
- **Polling:** the master asks each worker for its models and queue every 5 s. A worker that has not answered for
  15 s gets no new work.
- **Routing:** a chat (`/v1/chat/completions`, `/v1/messages`, `/v1/responses`, `/v1/completions`) goes to a worker
  in two cases:
  - its `model` is not installed on the master but is on a worker. The worker with the model loaded goes first,
    then the one with the shortest queue.
  - the master has work waiting and a worker has the same model loaded and idle.
- **The answer:** streamed back through the master, with `x-strata-worker: <name>`, and kept in the master's API
  call records.
- **Failure:** a worker that does not take a request leaves it to the master when the master has the model;
  otherwise the request is refused plainly (503, naming the worker).
- **Authentication:** only a request with the master's own key is sent on, never the client's key. The worker gets
  the worker's key.
- **The status:** `GET /api/cluster` shows each worker's models, its loaded model, its queue, when it was last seen,
  and how many requests it has taken.
- **Security for now:** plain HTTP with the key, so across a network you don't trust, use your own VPN
  (Tailscale, WireGuard). Paired ed25519 keys, encryption and mDNS discovery are the next steps (parts 1 and 2
  below).
- **Tested:** `a_master_routes_a_chat_to_the_worker_that_has_the_model` (two servers on one machine: a worker's
  model answered through the master, streamed too; the master's own model answered locally; no key, not sent on),
  and the routing rules in unit tests.

## What a cluster can and cannot buy (the physics first)

- **A model too big for one machine runs at all.** Its layers are split across machines (pipeline parallelism): each
  machine holds some layers, and the activations go from one to the next. This is what exo and llama.cpp's RPC
  backend do.
- **More requests at once.** Different machines answer different requests, each with a model it holds (request
  routing). This is the quickest win: no traffic between machines inside a request.
- **Longer context.** The KV cache is split across machines, by layers (each machine keeps the KV of its own layers)
  or by sequence (ring attention). More memory in all means more context.
- **Not a faster single answer over plain Ethernet.** A token must pass every machine in turn, so pipeline splitting
  adds one network hop per machine per token. About 0.2-1 ms on gigabit Ethernet; Wi-Fi is worse. Tensor parallelism
  (splitting each layer) needs far more bandwidth than home networks have. It is left out except over Thunderbolt /
  USB4 links, measured first.
- **Old phones** hold little RAM (2-8 GB) and slow memory. They are useful for small jobs:
  - a small model of their own;
  - a draft model for speculative decoding;
  - embeddings;
  - speech to text;
  - a few layers of a big model, where the network allows.

## The parts

1. **Nodes and joining.**
   - Every Prastara can be a node: `"cluster": {"role": "node"}`, or the Settings › Cluster page.
   - **On the same network**, nodes find each other by mDNS (`_am2strata._tcp`). A new node shows as "wants to join"
     until the user accepts it on the main machine. Nothing joins by itself.
   - **Remote machines** pair with keys: each node has an ed25519 key pair; pairing exchanges public keys once, by a
     code shown on one machine and typed on the other, or a QR code for a phone.
   - Traffic is end-to-end encrypted and mutually authenticated (Noise or TLS with the pinned keys).
   - Across the internet: a direct connection when the routers allow it (hole punching); otherwise through the
     user's own VPN (Tailscale, WireGuard), which the plan supports first because it is simplest and safe.
   - No third-party relay by default.
2. **What each node reports:**
   - its CPU, GPUs, RAM, VRAM and free disk;
   - the network's speed to each other node, measured with a short test;
   - the models it holds;
   - its battery and temperature (phones);
   - whether it is busy.
3. **Request routing (phase 1, the first to build).**
   - The main machine's queue (queue.rs) sees every node's models and queue.
   - A request goes to the node that holds its model and will answer soonest, by the same estimates the queue
     already makes. Media jobs go to a node with a free GPU.
   - The client still talks to the main machine only, and the answer is streamed back through it.
4. **Layers across machines (phase 2).**
   - A model too big for one node is split by layers, in proportion to each node's free memory and speed, as exo
     plans its partitions.
   - The engines speak a small protocol: activations in, activations out, KV kept where its layers are.
   - The first version uses llama.cpp's RPC backend as the measured reference: our engines are built on llama.cpp,
     which has `rpc-server`.
   - Strata's own engines get the same split after that, gated token for token against one-machine answers, as every
     engine change is.
   - MoE models (Tiel, Orca, the Qwen3 MoEs) can also split their experts: each node holds a share of the experts,
     and a token's router sends it to the nodes holding the experts it picked. This suits Strata's expert placement
     and is measured against plain layer splitting.
   **Phase 2, revised: memory across machines (owner, 2026-11-15).** The owner's rule: the same logic as one
   machine (the hottest experts in VRAM, the next in RAM, the rest read from the NVMe), with a new tier, **another
   machine's memory, over the network**. It is live resource sharing, not a demo.

   **Measured first (2026-11-15, the Linux PC and a Mac mini M4 worker):**

   | | |
   |---|---|
   | Round trip on the LAN, direct (the PC on 2.5 GbE) | **0.33 ms** (0.24-0.43) |
   | Round trip through Tailscale | ~1.7 ms, so the expert traffic uses the direct LAN address, still inside the pinned TLS |
   | This PC's memory | 59 GB RAM + 2 x 16 GB VRAM |
   | Models bigger than that, which read experts from the NVMe today | the 80 GB and 71 GB IQ3_XXS MoEs |
   | The Mac's free memory for experts | about 11 of its 16 GB |

   **Where a remote tier pays, and where it doesn't:**
   - **It pays where an expert would otherwise come from the NVMe.** A remote expert costs one round trip (about
     0.33 ms) plus the worker's compute. That is cheaper than reading the expert from the NVMe for each token.
   - **It is slower for a model that fits.** For a model that already fits in this PC's VRAM and RAM (Tiel 35B, 37 GB),
     every remote layer adds a round trip. So it is used only when an expert would otherwise be read from the NVMe.
   - **The first target is the 80 GB MoE.** With the Mac's 11 GB beside this PC's ~75 GB usable, the whole model fits
     in memory across the two machines.

   **The steps** (each gated by measurement and by token-for-token parity, as every engine change is):
   0. **Measure the rest:**
      - the link's bandwidth;
      - the Mac's expert compute for each weight format (IQ3_XXS, Q4_K, Q8_0), on its CPU and its GPU;
      - for the 80 GB model on this PC: today's speed and the share of time spent waiting on the NVMe.

      **Rule:** an expert goes remote only when the round trip plus the remote compute beats reading it from the
      NVMe.
   1. **An expert host on the worker** (Rust, `am2strata-hybrid --serve-experts`).
      - The worker's Prastara starts it when its master asks.
      - The master sends each assigned expert's tensors once over the pinned-TLS channel. The worker keeps them in RAM,
        cached on its disk by the model's hash.
      - It answers batched requests: {layer, the tokens' hidden states, each token's experts and weights} → their
        weighted sum.
      - **Gate:** its answers equal the master's own computation of the same experts, within the engine parity
        tolerance.
   2. **Placement across tiers:**
      - The master's engine fills VRAM → RAM → each worker's free memory (from its report) → NVMe, by the expert
        profile.
      - Remote memory takes the next-hottest experts after RAM, not the coldest, because it beats the NVMe.
      - **Needs first:** the worker reporting its real free memory. macOS reads 0 today, and that fix is queued.
   3. **The engine on the master:**
      - Per layer, the routed experts held remotely go out as one request per worker, while the local experts compute.
        Their result is added in.
      - A prompt batch makes one round trip per layer for all its tokens.
      - **Gates:** teacher-forced token-for-token parity with and without the remote tier; decode and prompt speed
        against the NVMe baseline.
   4. **Failure:** a worker that stops answering mid-answer has its experts read from the NVMe again. No wrong token,
      and the answer goes on. Tested by killing the host while it streams.
   5. **Shown and documented:**
      - Settings › Cluster shows what each worker holds (for example "11 GB of experts for the 80 GB model on macm4").
      - Monitor shows the remote share and its latency.
      - docs/CLUSTER.md gets the setup.
   6. **Then layer splitting** (whole layers on a worker) as a second mode, for models where it costs fewer round
      trips. Then the same tier in the Rust engine on the master.

   **Built in the Rust engine (2026-11-16, owner: "Mac and PC combo, Mac to Mac and PC to PC"):** steps 1, 3 and 4 and
   the engine half of step 2 (`crates/strata-hybrid/src/remote.rs`). The tier works in any direction: whichever machine
   runs the model is the master, so the 16 GB Mac running a 17.5 GB model can use the PC's RAM instead of its SSD. The
   gate is stricter than planned: identical tokens, because the integer kernels give the same bits on every CPU and the
   master adds each expert's output in routing order. Measured in docs/HYBRID-SPEED.md. **Still to build (stage B):**
   the servers starting the host and the master's flags from the pairing data (node-tls.json, the pins already
   swapped), the worker's report of the memory it lends, and Settings › Cluster and Monitor showing the placement.

   **Step 0, measured (2026-11-16): on this PC a remote expert tier would not pay today.**
   - Orca (the 80 GB IQ3_XXS MoE, the first target), loaded on the live server, same 2,174-token chat twice:
     - prompt 266 tok/s, generation 67 tok/s; then, with the prompt cached, generation 85 tok/s;
     - the engine's own counters while generating: 98.2% of expert lookups hit the GPU cache, 37 GB of experts
       resident in RAM, and **0 expert reads from the file**. Only the first request after loading read from storage,
       while the cache warmed.
   - This PC's 59 GB of RAM and 32 GB of VRAM hold almost all of an 80 GB model. So the NVMe is not the bottleneck
     the remote tier was meant to replace, and every remote hop would only add latency.
   - **When it would pay:** for a model clearly bigger than this PC's ~85-90 GB of RAM and VRAM, and with more remote
     memory than the Mac's ~11 GB (that gives only ~10% more room).
   - **Decision:** keep steps 1-6 as the design. Build them when such a model is wanted and a worker with the memory
     for it is in the cluster (a 64 GB+ machine, for example).
   - **The cluster pays today in other ways:** small models on workers as experts ("several brains", live since
     2026-11-16) and whole requests routed to workers.

   **Media jobs on workers (owner, 2026-11-16): renders shared across machines.**
   - **The ask:** video rendering run by Prastara across the cluster, with the Mac helping, before upload.
   - **Today:** a media job (image, video, compose, speech) runs on the machine that received it; only chats are routed
     to workers. Promo videos are rendered by scripts outside Prastara.
   - **Plan:**
     - `/v1/media/videos/compose` splits a render into independent segments;
     - the master sends a share of them to ready workers over the pinned cluster link (worker form, like experts), joins
       the results, and keeps any segment a worker fails;
     - a worker encodes with its hardware encoder when it has one: VideoToolbox (`h264_videotoolbox`) on Apple
       silicon, NVENC on NVIDIA; else libx264.
     - then the demo pipeline renders through compose, so videos made for the channel use the cluster.
   - **Baseline to beat:** the browser-use demo (2:47, 2560x1440, 11 segments) took 138 s with libx264 on this PC's
     CPU alone.
   - **Gates:**
     - the joined video equals a single-machine render frame for frame, in count, duration and audio sync;
     - the time is measured against the baseline;
     - a worker that drops mid-render has its segments done locally.

   **Both sides know where every part of a model is (owner, 2026-11-15).**
   - The master's engine and each worker share one placement map: for each model, which layers and experts sit in
     which machine's VRAM, RAM or disk.
   - A request is planned with that map, so the work for each part goes where that part already is, and nothing is
     loaded twice.
   - Each worker reports its map with its health every few seconds. The master's Settings › Cluster and Monitor show
     it.

   **Several brains: small models on the workers as experts (owner, 2026-11-15).**
   - A worker also keeps one or more small models of its own: for example an 8B MoE with 1B active, a math model or a
     coder.
   - The main model consults them in the same turn through `ask_expert`, and they get the user's original request.
   - **Cluster-aware experts:** an expert names a paired worker and its model (`{"name": "mac-small", "worker": "macm4",
     "model": "..."}`), instead of a URL and a key.
     - The call goes over the pinned-TLS cluster channel, with the key that pairing gave.
     - The expert is offered only while that worker is ready.
     - The choice of models follows the worker's measured speed. On the Mac: the GPU test decides whether its CPU or
       its GPU runs them.

5. **Long context across machines (phase 3).** The KV of each node's layers stays on that node (it comes free with
   phase 2). Sequence splitting (ring attention) for contexts beyond what layer splitting gives is kept for later,
   measured first.
6. **Phones (phase 4).**
   - A small node app: the same Rust engine built for Android (aarch64, by Termux first, an app later). It runs
     what fits, and phase 2 layers when the network allows.
   - Battery and temperature limits: a phone leaves the cluster when it is low on battery or hot, and its work goes
     back to the others.
7. **Failure.**
   - A node that stops answering is dropped from the routing at once.
   - A request it was running is told and retried on another node when one holds the model; for a split model the
     split is rebuilt without it, or the request is refused plainly when the model no longer fits.
   - The PRODUCTION-PLAN rules hold across the cluster: no crash, and overload told plainly.

## Master and workers (owner, 2026-11-11)

The cluster has one **master** and any number of **workers**:
- **The master:**
  - is the Prastara people and apps talk to; it holds the API keys, client profiles, memory, MCP servers and the
    queue;
  - routes each request to a worker (phase 1) or splits a model across workers (phase 2);
  - streams the answer back.
- **A worker:**
  - runs engines and media models only;
  - takes work from its master and from no one else; its own API is closed by default (`"cluster": {"role":
    "worker", "master": "<master's key fingerprint>"}`);
  - reports its models, free memory, queue and health every few seconds.
- **Joining:** a worker joins by pairing (a code typed on the master, or a QR code for a phone). The master can
  remove it at any time.
- **Failover:**
  - a worker that stops reporting is drained (no new work) and, after a timeout, dropped; its running requests
    are retried on another worker that holds the model, or refused plainly;
  - when the master itself goes down, workers wait; they do not take over, so there is never a second master.
    A standby master is a later step.
- **Peer to peer:**
  - on the same network, workers find the master by mDNS;
  - across the internet, through the user's own VPN (Tailscale, WireGuard) first; direct connections by hole
    punching later;
  - traffic is always end-to-end encrypted with the paired keys.

## Security

- **Encryption with the paired keys (built 2026-11-11, `crates/am2strata/src/cluster_tls.rs`):** mutual TLS 1.3.
  - Every node has a self-signed certificate in `<data>/node-tls.json` (0600). Its SHA-256 is its pin.
  - Pairing swaps the pins over the same request the 6-digit code proves: the master sends its pin, and the worker
    answers with its own and its TLS port. Each side keeps the other's pin (`cluster-workers.json` /
    `cluster-masters.json`, 0600).
  - A worker serves a second listener for its masters (`"cluster": {"role": "worker", "tls_port": N}`, default the
    API port + 443): TLS 1.3 only, a client certificate required, and only a paired master's certificate is let in.
  - A master's key works only on a connection showing that same master's certificate. Plain HTTP with it is refused
    unless the worker sets `"allow_plain": true` (masters paired before TLS keep working over plain HTTP).
  - The master reaches the worker only over that listener, checking the worker's certificate against its pin (no
    certificate authority: the pin is the trust), and shows its own certificate.
  - Why TLS and not Noise: rustls is already in the build (reqwest), audited and fast, and HTTP and streaming run over
    it unchanged.
  - Tested: `a_paired_worker_is_reached_only_over_pinned_tls` (two servers on one machine), plus a unit test with real
    handshakes (`only_pinned_certificates_talk`):
    - the master's key over plain HTTP: 401;
    - the TLS listener without a client certificate: refused at the handshake, as is an unpaired certificate;
    - a worker showing another certificate is refused by the master, which reports "the worker's certificate is not
      the one paired";
    - chats, streamed too, go through.
  - Measured on this PC (mock engines, 40 short chats each): the median chat through the master took 1.0 ms over
    plain HTTP and 1.0-1.1 ms over TLS. Connections are reused, so TLS adds about 0.1 ms.
  - **Limits:**
    - the pairing request itself is plain HTTP, so pair on a network you trust or through a VPN (someone in the middle
      of that one request could swap the pins);
    - workers listed in the run config by url and key stay plain HTTP, shown as "Plain HTTP: use a VPN";
    - hole punching across the internet is not built.
- **Joining:** only machines the user accepted or paired, never by being on the network alone.
- **Keys:** a node's keys never leave it. The main machine's API keys and client profiles stay on the main machine.
- **Prompts and answers:** they leave the main machine only to the nodes doing the work, encrypted. The node list
  shows what was sent where (in API calls).
- **No programs:** a node never runs MCP servers, programs or browser sessions for another node. Only model work.

## Measurements before any claim

On this PC (2 x RTX 5060 Ti) and a second machine over gigabit Ethernet, then Wi-Fi, then a phone, for each phase:
- tokens a second;
- the time to the first token;
- the largest model and context that run;
- each against one machine alone.

The numbers go in this file before the feature is called done.

## Related work (to read, not to copy without checking each licence)

- exo (exo labs): automatic partitioning by memory, phones as nodes.
- llama.cpp's RPC backend (`rpc-server`): layers on remote machines; MIT licence, like the rest of llama.cpp.
- Petals: layers across volunteers' machines over the internet.
- Prior art on ring attention for long context.

## One machine (owner, 2026-11-19)

The owner's aim: the machines bond into **one larger machine**. Models, data, tools, compute, memory, storage and
caches are shared through Prastara's own pipeline. Each machine is the **main** one for the people using it, and the
others are its subordinates; this works both ways, so the PC uses the Mac and the Mac uses the PC. It must work with
**any number** of machines, not just two.

It goes in phases. Each phase is measured before anything is claimed (see "Measurements before any claim").

| # | Part | What it means | Status |
|---|---|---|---|
| A | Chat routing | a chat goes to the machine that has its model, or to an idle one when this one is busy | built (phase 1) |
| B | Media routing | images, video, music, speech, transcription, compose and thumbnails go to an idle machine that can make them; a compose comes back as a job of the main machine's own. Only to a machine with the memory for it: each worker's queue says its free RAM, each GPU's free VRAM (a Mac's unified memory as its RAM) and its resident text model's memory, which a GPU media job frees; under the kind's floor (music 8 GB, image 6, video 10, speech and transcription 3, compose and thumbnails 1; `"cluster": {"media_min_free_gb": {...}}` on the main machine overrides them) the job goes to another worker or stays here, with an `x-strata-note` saying why. A worker that runs out of memory anyway is answered as such (`worker_out_of_memory`), and the job made here once when this machine can (`"strata_fallback": false` keeps the error) | built, 1c3f457; memory floors 2026-11-27 |
| C | Tools travel with a request | a routed request keeps its caller's tool sets (`x-strata-routed-tools`); memory and MCP stay on the main machine | built, 1c3f457 |
| D | Both ways | each machine pairs with the other as its master too, so either can be the main one; the masters list and the workers list already allow it, and pairing in both directions is tested | next |
| E | Tools across machines | a tool that exists on one machine only (its browser, its desktop, an MCP server, a model) is called from the other over the pinned link: a `remote_tool` call that the receiving side runs for its master's key, with the results streamed back | next |
| F | Models and data shared | a model or media file on one machine is copied to another by Prastara itself: content-addressed (the media library ids are hashes already), resumable, over the pinned TLS link, checked by hash, so no machine downloads the same model twice from the internet | **partly built**: the files of a Workshop job travel this way (J); models for chat, and media files, not yet |
| J | Workshop jobs across the cluster | a Workshop job (clean, convert, quantize, datagen, regress, train, ...) runs on the machine the request names, or on the idlest one that can (`"node": "any"`); its inputs go there and its outputs come back by hash | built (below) |
| G | Caches shared | a prompt's KV prefix (the system prompt of an agent, a long document) made on one machine is reused by the other when it is cheaper to send than to recompute; embeddings and web-fetch caches are shared by key | after F |
| H | Compute and memory pooled | one model split by layers across machines (phase 2 above: pipeline parallel, KV kept where its layers are). This is what pools memory: a model bigger than one machine runs at all, and the context is longer | after F, measured against llama.cpp RPC |
| I | Storage pooled | a job's files are kept where they were made and read through the main machine (they already are, for media jobs); the free disk of each machine is shown and used by F | after F |

**What the physics allows.** These limits are honest and are shown in the app:
- Pooling **memory capacity** works: a larger model fits, and the context gets longer.
- Pooling **memory bandwidth** does not speed up a single answer over Ethernet or Wi-Fi. Each token crosses every
  machine in turn, so one answer is never faster than on the fastest single machine.
- **Throughput** does add up: different requests and different media jobs run at the same time on different
  machines. That is what parts A and B give today.
- Over a USB4 or Thunderbolt link, splitting layers across machines may pay off even for a single answer. That is
  measured first, not claimed.

### Workshop jobs across the cluster (owner, 2026-11-27: "make cluster group work"; `crates/am2strata/src/cluster_workshop.rs`)

So the Mula plan's data cleaning, packing, datagen, tests and training can use every machine.

- **The request.** `POST /v1/workshop/jobs` takes an optional `"node"`:
  - a paired worker's name (`"macm4"`);
  - `"any"`: the idlest machine that can run that kind, this one included (fewest Workshop jobs, then the most free
    RAM, then the most free disk);
  - left out: this machine, as before.

  `"keep_on_node": true` leaves the outputs on the worker. The answer is the master's own job (202, the same id space),
  with `"node": "macm4"`. When `"any"` picks this machine, the job is an ordinary local one, with `"picked"` saying so.
- **The job.** `GET /v1/workshop/jobs/{id}` on the master shows the worker's state, log (each line prefixed with the
  worker's name), progress, points and result, mirrored every second. Its result adds `"node"`, `"remote_output"` and
  `"transfer"`, which lists each file with its sha256, bytes, and whether it was cached or resumed (and at which
  byte). `DELETE` cancels it and forwards the cancel to the worker's job (within a second).
- **Files (part F, minimal).** The master checks the request with its own Workshop's rules first: the same
  refusals, the same default output, paths only inside its Workshop folders, and every file inside a model folder too
  (a link out of them is refused). Then:
  1. Each input is hashed (SHA-256, streamed; a file is hashed again only when its size or time changed). A worker
     that has the hash is sent nothing (`GET /api/cluster/files/{sha}`).
  2. Otherwise the file is streamed in 1 MiB chunks (`PUT /api/cluster/files/{sha}?offset=&size=`). The worker
     appends to `<workshop>/cluster/partial/<sha>.part`. A cut stream keeps what came, and the next try resumes at
     that byte. The file becomes a read-only blob (`cluster/blobs/<sha>`) only when all of it hashes right. There are
     six tries, with a growing pause between them.
  3. The job is sent with each path replaced by a placeholder. The worker makes its own paths: hard links of the blobs
     in `cluster/in/<hash>/<name>` (the names are kept), outputs in `cluster/out/<tag>/<name>`. A suite travels as its
     JSON, with its tests' data files sent too.
  4. When it is done, the outputs come back the same way: listed with their hashes, streamed into the path the master
     planned (`.cluster-part`, resumed after a cut, renamed only when the hash matches), and skipped when the master
     already has that hash there.

  Nothing is held whole in memory, so a 16 GB model folder streams. The worker answers these routes only for a
  master's key on its pinned link: not for its owner's key, and never for a client.
- **Capabilities** (`GET /api/cluster/workshop` on the master, `/api/cluster/workshop/caps` on a worker). For each
  node:
  - whether its `am2strata-workshop` is installed;
  - the kinds it runs, with the CPU kinds (clean, convert, quantize, datagen through the node's own chat API, regress
    on the CPU, strata, merge, loop);
  - its GPU: cuda when `nvidia-smi -L` lists one; Metal only when the run config says
    `"cluster": {"workshop_gpu": "metal"}`, for a Workshop built with `--features metal`; `"none"` turns it off;
  - its free RAM and disk, and its active jobs.

  `"any"` sends a GPU job (`devices` cuda, or GPU `engine_args`) only to a CUDA node. A datagen job that names a model
  goes only to a node that has it.
- **Privacy.** A job is private when an input or output is:
  - marked `prastara_private` / `private: true` in a config.json, or in a dataset row;
  - a model config with the `private`, `uncensored` or `abliterated` flags;
  - a GGUF marked private;
  - anything under `Strata-data/private`.

  A private job is routed only to a node the request names. `"any"` refuses it (400 `private_data`, naming the file
  and the row). Workshop jobs never go to a remote provider. The Workshop tab says so under "Run on".
- **CLI and UI.**
  - CLI: `am2strata-run workshop <kind> ... --node macm4|any [--keep-on-node]`.
  - The Workshop tab has a "Run on" bar above its forms (This machine, Any, each worker with its GPU and free RAM; a
    worker that cannot take jobs is greyed out). It is shown only when there are workers. Job cards carry a node
    badge.
- **Tested** (`tests/cluster_workshop.rs`, two servers on this machine paired over pinned TLS, a stand-in Workshop
  command on the worker, CPU only):
  - a clean job runs on the worker and its output comes back matching by hash;
  - the same input again is not re-sent;
  - a transfer cut at half resumes at exactly that byte;
  - a private dataset row is refused with `"any"` and taken with the worker's name;
  - `"any"` picks the worker when this machine cannot;
  - a cancel reaches the worker's job;
  - the worker's side refuses its own owner's key.
  - Unit tests: names and paths, the privacy marks, capability matching.
- **Honest limits.**
  - **Time.** A 16 GB model over gigabit Ethernet takes about 2-3 minutes; over Wi-Fi (200-400 Mbit/s in practice)
    6-12 minutes. Hashing it first takes about half a minute to a minute (once per file version). Copy a model once
    and keep it there: the hash cache makes later jobs on it free.
  - **Disk and memory.** The worker's blobs are never evicted yet: they stay in `<workshop>/cluster/blobs` until
    removed by hand. A blob and its hard links share the disk space. The worker's routed jobs are kept in memory: after
    it restarts, the master says "no longer knows its job" and fails the mirror plainly.
  - **Datagen.** A datagen job uses the worker's own chat API and model. A worker whose chat goes to a remote (assist)
    provider would send the sources there. Private datagen is not blocked on that yet: name a worker whose model is
    local.
  - **Not tested here.**
    - The resume of an output copy back (the same code path as the input side, but no test cuts it).
    - A real 16 GB transfer.
    - The real Mac.
  - **Not covered.** A model in a Hugging Face cache (its files are links outside the Workshop's folders) is refused:
    copy it into a Workshop folder first. There is no `pack` kind in the Workshop on this branch yet, so it is not
    routable.

## Native media pipeline (owner, 2026-11-19)

Compose and thumbnails need ffmpeg today, a tool to install on every machine. The aim: Prastara's media engine does
this itself, in Rust, so a new machine needs nothing else. Parts:
- **Decode:**
  - symphonia for audio (already in strata-media);
  - openh264 for H.264 clips (already there, bundled and built from source);
  - the `image` crate for pictures.
- **Pictures:** scale, crop and zoompan done in Rust (bilinear, sub-pixel), and crossfades as a per-pixel blend in
  yuv420p.
- **Captions and cards:** rustybuzz for shaping (Devanagari and Malayalam need it) and ab_glyph to draw the glyphs,
  with the fonts the app already ships plus Noto Devanagari/Malayalam. Fade in and out as libass is used today.
- **Encode:**
  - openh264 for H.264;
  - the `mp4` crate to mux (already there, faststart);
  - audio as Opus in MP4 (libopus is bundled and built from source; YouTube accepts it), with AAC-LC as a later step.
- **Gate:** the same inputs through ffmpeg and through the Rust path are checked frame by frame, with SSIM ≥ 0.98 on
  sampled frames, the same length within one frame, and audio within 1 ms. ffmpeg stays as the fallback until the
  gate passes on both the PC and the Mac.
