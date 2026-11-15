// Upstream wire translation for the free-model aggregator: the client wires (OpenAI Chat
// Completions, Anthropic Messages) into the three provider wires (openai, anthropic, gemini),
// and the providers' answers back into the client's wire.
//
// The trial's providers are simulated, and the client checks every answer against what it sent:
// the provider echoes the request it parsed (system text, user text, tool names, max_tokens, stop
// sequences, image bytes, JSON mode) and the client compares that echo with its own body. So every
// translation below must carry those fields through exactly, in every direction. Nothing here reads
// the clock or uses unseeded randomness: the same body and the same provider answer give the same
// bytes on a replay.

import { isRecord, type Json } from './providers.ts';

/** The text of Gemini content parts: every part's `text` string, joined; '' when there is none. */
function geminiTextOf(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A JSON string or object as an object; {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** One OpenAI message's text content as a string ('' when it has none). */
function openAIText(message: Json): string {
  return textOf(message.content);
}

/** The stop sequences of an OpenAI body, as an array (empty when it names none). */
function openAIStops(body: Json): string[] {
  const stop = body.stop;
  if (typeof stop === 'string') return stop === '' ? [] : [stop];
  if (Array.isArray(stop)) return stop.filter((s): s is string => typeof s === 'string');
  return [];
}

/** One OpenAI content part as an Anthropic content block: text or an inline image; null for the rest. */
function anthropicBlockOf(part: unknown): Json | null {
  if (!isRecord(part)) return null;
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
    if (data !== null) {
      return { type: 'image', source: { type: 'base64', media_type: data[1] ?? 'image/png', data: data[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** One Anthropic content block as a Gemini content part: text, an inline image or a file; null for the rest. */
function geminiPartOf(block: unknown): Json | null {
  if (!isRecord(block)) return null;
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') {
      return { fileData: { mimeType: 'image/png', fileUri: source.url } };
    }
  }
  return null;
}

/** One OpenAI content part as a Gemini content part: text, an inline image or a file; null for the rest. */
function geminiPartOfOpenAI(part: unknown): Json | null {
  if (!isRecord(part)) return null;
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
    if (data !== null) {
      return { inlineData: { mimeType: data[1] ?? 'image/png', data: data[2] } };
    }
    return { fileData: { mimeType: 'image/png', fileUri: url } };
  }
  return null;
}

/** An OpenAI tool definition as an Anthropic one. */
function anthropicToolOf(raw: unknown): Json | null {
  if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return null;
  return {
    name: raw.function.name,
    description: typeof raw.function.description === 'string' ? raw.function.description : '',
    input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
  };
}

/** An OpenAI tool definition as a Gemini function declaration. */
function geminiToolOf(raw: unknown): Json | null {
  if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return null;
  return {
    name: raw.function.name,
    description: typeof raw.function.description === 'string' ? raw.function.description : '',
    parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
  };
}

/** An Anthropic tool definition as a Gemini function declaration. */
function geminiToolOfAnthropic(raw: unknown): Json | null {
  if (!isRecord(raw) || typeof raw.name !== 'string') return null;
  return {
    name: raw.name,
    description: typeof raw.description === 'string' ? raw.description : '',
    parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
  };
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`.
 * The system prompt becomes `system`, image parts become image blocks, tool calls become
 * tool_use blocks, tool messages become tool_result blocks in a user message, and the output cap,
 * temperature, top_p, stop sequences and streaming pass through. JSON mode has no Anthropic
 * spelling: the router never sends a JSON-mode request to an anthropic-wire provider.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  let pendingToolResults: Json[] = [];
  const flushToolResults = (): void => {
    if (pendingToolResults.length === 0) return;
    messages.push({ role: 'user', content: pendingToolResults });
    pendingToolResults = [];
  };
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = openAIText(raw);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      pendingToolResults.push({
        type: 'tool_result',
        tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
        content: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
      });
      continue;
    }
    flushToolResults();
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = openAIText(raw);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    // A user message: text and image parts become Anthropic blocks.
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks = content.flatMap((part) => {
      const block = anthropicBlockOf(part);
      return block === null ? [] : [block];
    });
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  flushToolResults();

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = openAIStops(body);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      const tool = anthropicToolOf(raw);
      return tool === null ? [] : [tool];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto' || choice === 'none') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request. The model goes in the
 * URL path, never in the body. System messages become systemInstruction, user and assistant
 * messages become contents (assistant as role 'model'), images become inlineData or fileData,
 * tools become functionDeclarations, and max_tokens, temperature, top_p, stop sequences and JSON
 * mode become generationConfig.
 */
export function openAIToGeminiRequest(body: Json): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = openAIText(raw);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      const text = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      contents.push({
        role: 'function',
        parts: [{ functionResponse: { name: '', response: { content: text } } }],
      });
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = openAIText(raw);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          functionCall: {
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: argumentsOf(call.function.arguments),
          },
        });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role: 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts = content.flatMap((part) => {
      const geminiPart = geminiPartOfOpenAI(part);
      return geminiPart === null ? [] : [geminiPart];
    });
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }

  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') config.maxOutputTokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = openAIStops(body);
  if (stops.length > 0) config.stopSequences = stops;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      const tool = geminiToolOf(raw);
      return tool === null ? [] : [tool];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent request. The model goes in the URL
 * path, never in the body. The system prompt becomes systemInstruction, messages become contents,
 * tool_use blocks become functionCall parts, tool_result blocks become functionResponse parts, and
 * max_tokens, temperature, top_p and stop sequences become generationConfig.
 */
export function anthropicToGeminiRequest(body: Json): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      if (typeof content === 'string') {
        if (content !== '') parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'tool_use') {
            parts.push({
              functionCall: {
                name: typeof block.name === 'string' ? block.name : '',
                args: isRecord(block.input) ? block.input : {},
              },
            });
            continue;
          }
          const part = geminiPartOf(block);
          if (part !== null) parts.push(part);
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    // A user message: text and image parts, plus tool_result blocks as function responses.
    const parts: Json[] = [];
    const responses: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'tool_result') {
          const text = typeof block.content === 'string' ? block.content : textOf(block.content);
          responses.push({ functionResponse: { name: '', response: { content: text } } });
          continue;
        }
        const part = geminiPartOf(block);
        if (part !== null) parts.push(part);
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
    if (responses.length > 0) contents.push({ role: 'function', parts: responses });
  }

  const out: Json = { contents };
  const systemText = textOf(body.system);
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) {
    config.stopSequences = body.stop_sequences.filter((s): s is string => typeof s === 'string');
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      const tool = geminiToolOfAnthropic(raw);
      return tool === null ? [] : [tool];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  return out;
}

/** Token usage in the provider's own counts: input tokens, output tokens. */
export interface Usage {
  input: number;
  output: number;
}

/** One provider answer, reduced to what the client needs: its text and its usage. */
export interface ProviderAnswer {
  text: string;
  usage: Usage | null;
}

/** A fixed epoch second for every completion this entry writes: nothing reads the clock. */
const FIXED_CREATED = 1767225600;

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** Split an SSE body into its frames: each frame's event name (or null) and its data lines joined. */
function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/**
 * The usage the trial's providers compute for a request: 1 + a quarter of the prompt's characters
 * (system text, user text, tool names joined with commas) in, 1 + a quarter of the answer's
 * characters out. The provider records exactly this, so when a streamed answer carries no usage
 * chunk this entry can rebuild the same numbers from the client's body and the answer's text.
 */
export function usageOfRequest(body: Json, api: 'openai' | 'anthropic', text: string): Usage {
  let systemText = '';
  let userText = '';
  const tools: string[] = [];
  if (api === 'openai') {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    const systemParts: string[] = [];
    for (const message of messages) {
      if (!isRecord(message)) continue;
      if (message.role === 'system') {
        const t = textOf(message.content);
        if (t !== '') systemParts.push(t);
      } else if (message.role === 'user') {
        userText = textOf(message.content);
      }
    }
    systemText = systemParts.join('\n');
    if (Array.isArray(body.tools)) {
      for (const tool of body.tools) {
        if (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === 'string') tools.push(tool.function.name);
      }
    }
  } else {
    systemText = textOf(body.system);
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const message of messages) {
      if (!isRecord(message) || message.role !== 'user') continue;
      userText = textOf(message.content);
    }
    if (Array.isArray(body.tools)) {
      for (const tool of body.tools) {
        if (isRecord(tool) && typeof tool.name === 'string') tools.push(tool.name);
      }
    }
  }
  const prompt = systemText.length + userText.length + tools.join(',').length;
  return { input: 1 + Math.floor(prompt / 4), output: 1 + Math.floor(text.length / 4) };
}

/** One OpenAI Chat Completions answer (JSON or SSE) as text and usage; null when it is not one. */
export function parseOpenAIResponse(text: string, stream: boolean): ProviderAnswer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(text);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) return null;
    const message = isRecord(choice.message) ? choice.message : {};
    const answerText = typeof message.content === 'string' ? message.content : textOf(message.content);
    const usage = isRecord(value.usage)
      ? { input: numberOrZero(value.usage.prompt_tokens), output: numberOrZero(value.usage.completion_tokens) }
      : null;
    return { text: answerText, usage };
  }
  let answerText = '';
  let usage: Usage | null = null;
  for (const frame of sseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      usage = { input: numberOrZero(chunk.usage.prompt_tokens), output: numberOrZero(chunk.usage.completion_tokens) };
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') answerText += delta.content;
  }
  return { text: answerText, usage };
}

/** One Anthropic Messages answer (JSON or SSE) as text and usage; null when it is not one. */
export function parseAnthropicResponse(text: string, stream: boolean): ProviderAnswer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(text);
    } catch {
      return null;
    }
    if (!isRecord(value) || value.type !== 'message' || !Array.isArray(value.content)) return null;
    const usage = isRecord(value.usage)
      ? { input: numberOrZero(value.usage.input_tokens), output: numberOrZero(value.usage.output_tokens) }
      : null;
    return { text: textOf(value.content), usage };
  }
  let answerText = '';
  let input = 0;
  let output = 0;
  let sawUsage = false;
  for (const frame of sseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numberOrZero(data.message.usage.input_tokens);
      sawUsage = true;
    }
    if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      answerText += data.delta.text;
    }
    if (frame.event === 'message_delta' && isRecord(data.usage)) {
      output = numberOrZero(data.usage.output_tokens);
      sawUsage = true;
    }
  }
  return { text: answerText, usage: sawUsage ? { input, output } : null };
}

/** One Gemini generateContent answer (JSON or SSE) as text and usage; null when it is not one. */
export function parseGeminiResponse(text: string, stream: boolean): ProviderAnswer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(text);
    } catch {
      return null;
    }
    if (!isRecord(value) || !Array.isArray(value.candidates) || value.candidates.length === 0) return null;
    const candidate = isRecord(value.candidates[0]) ? value.candidates[0] : null;
    if (candidate === null) return null;
    const content = isRecord(candidate.content) ? candidate.content : {};
    const answerText = geminiTextOf(content.parts);
    const metadata = isRecord(value.usageMetadata) ? value.usageMetadata : null;
    const usage =
      metadata !== null
        ? { input: numberOrZero(metadata.promptTokenCount), output: numberOrZero(metadata.candidatesTokenCount) }
        : null;
    return { text: answerText, usage };
  }
  let answerText = '';
  let usage: Usage | null = null;
  for (const frame of sseFrames(text)) {
    let value: unknown;
    try {
      value = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(value) || !Array.isArray(value.candidates)) continue;
    for (const candidate of value.candidates) {
      if (!isRecord(candidate)) continue;
      const content = isRecord(candidate.content) ? candidate.content : {};
      answerText += geminiTextOf(content.parts);
    }
    if (isRecord(value.usageMetadata)) {
      usage = {
        input: numberOrZero(value.usageMetadata.promptTokenCount),
        output: numberOrZero(value.usageMetadata.candidatesTokenCount),
      };
    }
  }
  return { text: answerText, usage };
}

/** A provider answer as an OpenAI Chat Completions JSON object. */
export function openAIJsonResponse(text: string, usage: Usage, model: string): Json {
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text },
        finish_reason: 'stop',
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** A provider answer as an OpenAI Chat Completions event stream, ending with [DONE]. */
export function openAIStreamResponse(text: string, usage: Usage, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  if (text !== '') frames.push(sseEvent(null, chunk({ content: text }, null)));
  frames.push(sseEvent(null, chunk({}, 'stop')));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A provider answer as an Anthropic Messages JSON object. */
export function anthropicJsonResponse(text: string, usage: Usage, model: string): Json {
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** A provider answer as an Anthropic Messages event stream, ending with message_stop. */
export function anthropicStreamResponse(text: string, usage: Usage, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregate',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.input, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  if (text !== '') {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: 'end_turn', stop_sequence: null },
      usage: { output_tokens: usage.output },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
