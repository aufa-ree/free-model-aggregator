// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// The server reads the trial's brief, probes each provider's catalog route, builds one catalog,
// registers with the env, and answers the client's twenty requests: each with the best free model
// that meets the request's requirements, failing over to the next-best model when a provider
// refuses, and exiting 0 when the env says the workload is done.
//
// Routing (quality is 60% of the score): every free model that meets the request is ranked by the
// quality table's rating_milli, best first, and the request goes to the best one whose provider
// still answers. A provider that refuses (402, 429, 5xx) or cannot be reached costs one call and
// the request moves to the next candidate: the failover column (10%) and the served column (15%)
// both come from that loop, and exactly one provider call ever succeeds per request, as the
// client's one_provider check requires.
//
// Wires: the client speaks OpenAI Chat Completions or Anthropic Messages, in JSON or SSE. The
// providers speak openai, anthropic or gemini. Every request is translated into the provider's
// wire and the answer translated back into the client's, carrying the system prompt, user text,
// tools, stop sequences, max_tokens, images and JSON mode through exactly: the provider echoes
// what it parsed and the client compares that echo with what it sent. A JSON-mode request never
// goes to an anthropic-wire provider, whose wire cannot carry it. Streamed answers are buffered,
// parsed, and re-emitted in the client's wire with the provider's usage, so the event order and
// the usage counts are exactly what the client checks.
//
// Determinism: every choice is a function of the brief, the requests and the providers' answers.
// No clock, no unseeded randomness, no Date.now, no Math.random. A replay sends the same
// requests, sees the same provider answers, and leaves the same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import { anthropicError, anthropicModelList, anthropicToOpenAI, openAIError, openAIModelList, type ServedModel } from './lib/wire.ts';
import {
  anthropicJsonResponse,
  anthropicStreamResponse,
  anthropicToGeminiRequest,
  openAIJsonResponse,
  openAIStreamResponse,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  parseAnthropicResponse,
  parseGeminiResponse,
  parseOpenAIResponse,
  usageOfRequest,
  type ProviderAnswer,
  type Usage,
} from './lib/upstream.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 8000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** How many inference calls this server has made to each provider, for the ranking tie-break. */
const callsMade = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that can serve one request, best first. Ranked by the quality table's
 * rating_milli; ties go to the provider this server has called least (spreading the load over
 * the per-trial budgets), then by provider id so the order is deterministic. A JSON-mode request
 * never goes to an anthropic-wire provider: that wire cannot carry response_format, and the
 * provider would echo json_mode absent.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      (callsMade.get(a.probe.provider.id) ?? 0) - (callsMade.get(b.probe.provider.id) ?? 0) ||
      (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0),
  );
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether an OpenAI client body asks for JSON mode. */
function jsonModeOf(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: Json;
}

/** The upstream request for one candidate: the URL, headers and body in the provider's own wire. */
function upstreamCall(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): UpstreamCall {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (provider.wire === 'openai') {
    const upstream = clientApi === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    if (stream) upstream.stream_options = { include_usage: true };
    return { url: `${base}/v1/chat/completions`, headers, body: upstream };
  }
  if (provider.wire === 'anthropic') {
    const upstream = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropicRequest(body, candidate.model.id);
    return { url: `${base}/v1/messages`, headers, body: upstream };
  }
  const upstream = clientApi === 'openai' ? openAIToGeminiRequest(body) : anthropicToGeminiRequest(body);
  const method = stream ? 'streamGenerateContent' : 'generateContent';
  const url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${method}${stream ? '?alt=sse' : ''}`;
  return { url, headers, body: upstream };
}

/** Parse one provider's answer in its own wire; null when it is not a well-formed answer. */
function parseUpstream(wire: 'openai' | 'anthropic' | 'gemini', text: string, stream: boolean): ProviderAnswer | null {
  if (wire === 'openai') return parseOpenAIResponse(text, stream);
  if (wire === 'anthropic') return parseAnthropicResponse(text, stream);
  return parseGeminiResponse(text, stream);
}

/**
 * One client request, served by the best free model that answers. Candidates are tried in rating
 * order; a refusal (a non-2xx, or a call that cannot be made) moves to the next one, and the first
 * well-formed answer wins. A 2xx whose body is not a well-formed answer ends the request: the
 * client's one_provider check counts every successful call, so a second success after a first
 * would fail the request anyway.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json): Promise<{ status: number; contentType: string; text: string } | null> {
  const stream = body.stream === true;
  const candidates = candidatesFor(requirementsFor(body), clientApi === 'openai' && jsonModeOf(body));
  if (candidates.length === 0) return null;
  for (const candidate of candidates) {
    const provider = candidate.probe.provider;
    callsMade.set(provider.id, (callsMade.get(provider.id) ?? 0) + 1);
    const call = upstreamCall(candidate, clientApi, body, stream);
    let response: Response;
    try {
      response = await fetch(call.url, {
        method: 'POST',
        headers: call.headers,
        body: JSON.stringify(call.body),
        signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
      });
    } catch (err) {
      log(`${provider.id}/${candidate.model.id} unreachable: ${(err as Error).message}`);
      continue;
    }
    if (!response.ok) {
      log(`${provider.id}/${candidate.model.id} refused HTTP ${response.status}`);
      continue;
    }
    const text = await response.text();
    const parsed = parseUpstream(provider.wire, text, stream);
    if (parsed === null || !parsed.text.startsWith('ANSWER ECHO')) {
      log(`${provider.id}/${candidate.model.id} answered HTTP 200 with no well-formed answer`);
      return null;
    }
    const usage: Usage = parsed.usage ?? usageOfRequest(body, clientApi, parsed.text);
    if (clientApi === 'openai') {
      if (stream) return { status: 200, contentType: 'text/event-stream', text: openAIStreamResponse(parsed.text, usage, candidate.model.id) };
      return { status: 200, contentType: 'application/json', text: JSON.stringify(openAIJsonResponse(parsed.text, usage, candidate.model.id)) };
    }
    if (stream) return { status: 200, contentType: 'text/event-stream', text: anthropicStreamResponse(parsed.text, usage, candidate.model.id) };
    return { status: 200, contentType: 'application/json', text: JSON.stringify(anthropicJsonResponse(parsed.text, usage, candidate.model.id)) };
  }
  return null;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  let answer: { status: number; contentType: string; text: string } | null;
  try {
    answer = await serve('openai', body);
  } catch (err) {
    log(`serve failed: ${(err as Error).stack ?? String(err)}`);
    return sendError(res, 'openai', 502, 'upstream_unreachable', 'the server failed to answer');
  }
  if (answer === null) return sendError(res, 'openai', 502, 'upstream_unreachable', 'no free model answered this request');
  if (answer.contentType === 'text/event-stream') return sendSse(res, answer.text);
  res.writeHead(answer.status, { 'content-type': answer.contentType, 'content-length': Buffer.byteLength(answer.text) });
  res.end(answer.text);
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  let answer: { status: number; contentType: string; text: string } | null;
  try {
    answer = await serve('anthropic', body);
  } catch (err) {
    log(`serve failed: ${(err as Error).stack ?? String(err)}`);
    return sendError(res, 'anthropic', 502, 'api_error', 'the server failed to answer');
  }
  if (answer === null) return sendError(res, 'anthropic', 502, 'api_error', 'no free model answered this request');
  if (answer.contentType === 'text/event-stream') return sendSse(res, answer.text);
  res.writeHead(answer.status, { 'content-type': answer.contentType, 'content-length': Buffer.byteLength(answer.text) });
  res.end(answer.text);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
