/**
 * 🪄 Universal AI Chat Module
 * ----------------------------------------------------------
 * Drop this single file into ANY Lovable project to add AI chat.
 *
 * ✅ Works with Lovable AI Gateway (Gemini + GPT-5 models)
 * ✅ Supports streaming (token-by-token) and non-streaming
 * ✅ Handles 429 (rate limit) + 402 (out of credits) gracefully
 * ✅ Optional structured JSON output via tool calling
 *
 * ----------------------------------------------------------
 * SETUP (one-time, per project)
 * ----------------------------------------------------------
 * 1. Enable Lovable Cloud in your project (this auto-provisions LOVABLE_API_KEY).
 * 2. Create an edge function `supabase/functions/ai-chat/index.ts` (see snippet below).
 * 3. Import this file anywhere:  import { chat, chatStream } from "@/lib/ai";
 *
 * ----------------------------------------------------------
 * EDGE FUNCTION — copy/paste into supabase/functions/ai-chat/index.ts
 * ----------------------------------------------------------
 *
 * import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
 *
 * const cors = {
 *   "Access-Control-Allow-Origin": "*",
 *   "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
 * };
 *
 * serve(async (req) => {
 *   if (req.method === "OPTIONS") return new Response(null, { headers: cors });
 *   try {
 *     const { messages, model, stream, tools, tool_choice } = await req.json();
 *     const KEY = Deno.env.get("LOVABLE_API_KEY");
 *     if (!KEY) throw new Error("LOVABLE_API_KEY missing");
 *
 *     const r = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
 *       method: "POST",
 *       headers: { Authorization: `Bearer ${KEY}`, "Content-Type": "application/json" },
 *       body: JSON.stringify({
 *         model: model ?? "google/gemini-3-flash-preview",
 *         messages,
 *         stream: !!stream,
 *         ...(tools ? { tools } : {}),
 *         ...(tool_choice ? { tool_choice } : {}),
 *       }),
 *     });
 *
 *     if (r.status === 429)
 *       return new Response(JSON.stringify({ error: "Rate limit, try again shortly." }),
 *         { status: 429, headers: { ...cors, "Content-Type": "application/json" } });
 *     if (r.status === 402)
 *       return new Response(JSON.stringify({ error: "Out of credits. Add funds in Settings → Workspace → Usage." }),
 *         { status: 402, headers: { ...cors, "Content-Type": "application/json" } });
 *     if (!r.ok)
 *       return new Response(JSON.stringify({ error: "AI gateway error" }),
 *         { status: 500, headers: { ...cors, "Content-Type": "application/json" } });
 *
 *     if (stream) {
 *       return new Response(r.body, { headers: { ...cors, "Content-Type": "text/event-stream" } });
 *     }
 *     const data = await r.json();
 *     return new Response(JSON.stringify(data), { headers: { ...cors, "Content-Type": "application/json" } });
 *   } catch (e) {
 *     return new Response(JSON.stringify({ error: e instanceof Error ? e.message : "Unknown" }),
 *       { status: 500, headers: { ...cors, "Content-Type": "application/json" } });
 *   }
 * });
 *
 * ----------------------------------------------------------
 */

export type ChatRole = "system" | "user" | "assistant";
export interface ChatMessage {
  role: ChatRole;
  content: string;
}

export type AIModel =
  | "google/gemini-3-flash-preview"
  | "google/gemini-3.1-pro-preview"
  | "google/gemini-2.5-pro"
  | "google/gemini-2.5-flash"
  | "google/gemini-2.5-flash-lite"
  | "openai/gpt-5"
  | "openai/gpt-5-mini"
  | "openai/gpt-5-nano"
  | "openai/gpt-5.2";

const DEFAULT_MODEL: AIModel = "google/gemini-3-flash-preview";

const ENDPOINT = `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/ai-chat`;
const AUTH = `Bearer ${import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY}`;

interface ChatOptions {
  model?: AIModel;
  system?: string;
  /** For structured output via tool calling */
  tool?: { name: string; description?: string; parameters: Record<string, unknown> };
}

function buildMessages(messages: ChatMessage[], system?: string): ChatMessage[] {
  return system ? [{ role: "system", content: system }, ...messages] : messages;
}

async function handleHttpError(res: Response): Promise<never> {
  let msg = `Request failed (${res.status})`;
  try {
    const j = await res.json();
    if (j?.error) msg = j.error;
  } catch {
    /* ignore */
  }
  if (res.status === 429) throw new Error("Rate limit reached. Please wait a moment.");
  if (res.status === 402) throw new Error("AI credits exhausted. Add funds in Settings → Workspace → Usage.");
  throw new Error(msg);
}

/**
 * 💬 Non-streaming chat. Returns the full assistant text.
 */
export async function chat(messages: ChatMessage[], options: ChatOptions = {}): Promise<string> {
  const res = await fetch(ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: AUTH },
    body: JSON.stringify({
      messages: buildMessages(messages, options.system),
      model: options.model ?? DEFAULT_MODEL,
      stream: false,
    }),
  });
  if (!res.ok) await handleHttpError(res);
  const data = await res.json();
  return data?.choices?.[0]?.message?.content ?? "";
}

/**
 * 🌊 Streaming chat. Calls onDelta for each token chunk.
 */
export async function chatStream(
  messages: ChatMessage[],
  onDelta: (chunk: string) => void,
  options: ChatOptions & { signal?: AbortSignal } = {},
): Promise<void> {
  const res = await fetch(ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: AUTH },
    body: JSON.stringify({
      messages: buildMessages(messages, options.system),
      model: options.model ?? DEFAULT_MODEL,
      stream: true,
    }),
    signal: options.signal,
  });
  if (!res.ok || !res.body) await handleHttpError(res);

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let done = false;

  while (!done) {
    const { value, done: streamDone } = await reader.read();
    if (streamDone) break;
    buffer += decoder.decode(value, { stream: true });

    let nl: number;
    while ((nl = buffer.indexOf("\n")) !== -1) {
      let line = buffer.slice(0, nl);
      buffer = buffer.slice(nl + 1);
      if (line.endsWith("\r")) line = line.slice(0, -1);
      if (!line || line.startsWith(":") || !line.startsWith("data: ")) continue;

      const json = line.slice(6).trim();
      if (json === "[DONE]") {
        done = true;
        break;
      }
      try {
        const parsed = JSON.parse(json);
        const delta = parsed?.choices?.[0]?.delta?.content;
        if (delta) onDelta(delta);
      } catch {
        // partial JSON — push back and wait for more
        buffer = line + "\n" + buffer;
        break;
      }
    }
  }
}

/**
 * 🧩 Structured output via tool calling. Returns parsed JSON matching your schema.
 *
 * Example:
 *   const data = await chatJSON<{ items: string[] }>(
 *     [{ role: "user", content: "List 3 fruits" }],
 *     {
 *       name: "list_fruits",
 *       parameters: {
 *         type: "object",
 *         properties: { items: { type: "array", items: { type: "string" } } },
 *         required: ["items"],
 *       },
 *     },
 *   );
 */
export async function chatJSON<T = unknown>(
  messages: ChatMessage[],
  tool: { name: string; description?: string; parameters: Record<string, unknown> },
  options: Omit<ChatOptions, "tool"> = {},
): Promise<T> {
  const res = await fetch(ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: AUTH },
    body: JSON.stringify({
      messages: buildMessages(messages, options.system),
      model: options.model ?? DEFAULT_MODEL,
      stream: false,
      tools: [{ type: "function", function: tool }],
      tool_choice: { type: "function", function: { name: tool.name } },
    }),
  });
  if (!res.ok) await handleHttpError(res);
  const data = await res.json();
  const args = data?.choices?.[0]?.message?.tool_calls?.[0]?.function?.arguments;
  if (!args) throw new Error("No structured output returned");
  return JSON.parse(args) as T;
}
