#!/usr/bin/env node
/**
 * Council Proxy Service
 * 
 * An OpenAI-compatible proxy that distributes requests to multiple LLM models
 * in parallel, then uses a Chairman model to fuse the best response.
 * 
 * Port: 3460 (localhost only)
 * Endpoint: POST /v1/chat/completions
 */

import { createServer } from 'node:http';
import { readFileSync, writeFileSync, appendFileSync, existsSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

// ─── Configuration ──────────────────────────────────────────────────────────

const PORT = 3460;
const HOST = '127.0.0.1';
const MEMBER_TIMEOUT_MS = 40_000;   // 40s per member (thinking models need more time)
const CHAIRMAN_TIMEOUT_MS = 45_000; // 45s for Chairman
const ROUTER_TIMEOUT_MS = 45_000;   // 10s 太短。多轮 tool call 累积后 messages 可能 20-40k tokens,DSV4 处理长 context 需要 5-15s,撞上 10s 必 timeout 失败。45s 对齐 CHAIRMAN_TIMEOUT_MS
const ROUTER_MAX_RETRIES = 3;       // max retries if tool_call validation fails
const ROUTER_MAX_TOKENS = 4000;     // 2000 仍不够。DSV4 在 round 1+ 累积 tool results 后生成多 tool_calls JSON 又被截断(position 1354 处)。4000 给充足余量,Router 输出 token cost 仍极小
const TOOL_CALL_MAX_ROUNDS = 10;    // max tool call rounds before returning status
const MIN_RESPONSES = 2;            // minimum members needed to proceed
const LOG_PATH = join(process.env.HOME || '/root', '.council-proxy', 'council-log.md');

// Single API key for OpenRouter (reuses existing OpenRouter key)
const OPENROUTER_KEY = process.env.OPENROUTER_API_KEY || '';
const OPENROUTER_BASE = 'https://openrouter.ai/api/v1';

//  add (): DeepSeek 直连
// OpenRouter 上的 V4 Flash 稳定 40s timeout(OpenRouter 路由/容量问题,非模型本身慢);
// 直连 curl 实测短问题 3s,议会场景预期 5-15s。key 复用 secrets.env 里早有的 DEEPSEEK_API_KEY。
const DEEPSEEK_KEY = process.env.DEEPSEEK_API_KEY || '';
const DEEPSEEK_BASE = 'https://api.deepseek.com/v1';

// ─── Council Members ────────────────────────────────────────────────────────

//  换血():B/E/F 加 enabled: false 保留历史代号不复用,新增 H/I
// 活跃阵容:A(Grok) / C(Qwen) / D(MiMo) / H(Gem3.1L) / I(DSV4)
const MEMBERS = [
  {
    id: 'A', name: 'Grok 4.1 Fast', shortName: 'Grok', provider: 'xAI',
    baseUrl: OPENROUTER_BASE,
    model: 'x-ai/grok-4.1-fast',
    extraBody: { reasoning: { effort: 'low' } },
    apiKey: () => OPENROUTER_KEY,
    supportsVision: true,
    contextLength: 2000000,
  },
  {
    id: 'B', name: 'Gemini 2.5 Flash', shortName: 'Gem2.5', provider: 'Google',
    baseUrl: OPENROUTER_BASE,
    model: 'google/gemini-2.5-flash',
    apiKey: () => OPENROUTER_KEY,
    supportsVision: true,
    contextLength: 1000000,
    extraBody: { reasoning: { max_tokens: 200 } },
    enabled: false, // 停用  ,被 H (Gemini 3.1 Flash Lite Preview) 替代
  },
  {
    id: 'C', name: 'Qwen3.5-Flash', shortName: 'Qwen', provider: 'Alibaba',
    baseUrl: OPENROUTER_BASE,
    model: 'qwen/qwen3.5-flash-02-23',
    extraBody: { reasoning: { max_tokens: 200 } },
    apiKey: () => OPENROUTER_KEY,
    supportsVision: true,
    contextLength: 1000000,
  },
  {
    id: 'D', name: 'MiMo-V2-Flash', shortName: 'MiMo', provider: 'Xiaomi',
    baseUrl: OPENROUTER_BASE,
    model: 'xiaomi/mimo-v2-flash',
    extraBody: { reasoning: { max_tokens: 500 } },
    apiKey: () => OPENROUTER_KEY,
    supportsVision: false, // text-only
    contextLength: 32000,
    enabled: false, // 临时禁用 :同一天连续 2 次 40s timeout(其他 4 人都 <10s),OpenRouter 上 MiMo 路由疑似容量问题。代号保留,几天后再启用观察
  },
  {
    id: 'E', name: 'Step 3.5 Flash', shortName: 'Step', provider: 'StepFun',
    baseUrl: OPENROUTER_BASE,
    model: 'stepfun/step-3.5-flash',
    extraBody: { reasoning: { max_tokens: 200 } },
    apiKey: () => OPENROUTER_KEY,
    supportsVision: false, // text-only, skip when images present
    contextLength: 200000,
    enabled: false, // 停用  ,被 I (DeepSeek V4 Flash) 替代(Step 响应长尾严重 p90=19.7s)
  },
  {
    id: 'F', name: 'GLM-4.7 Flash', shortName: 'GLM', provider: 'Z.ai',
    baseUrl: OPENROUTER_BASE,
    model: 'z-ai/glm-4.7-flash',
    extraBody: { reasoning: { enabled: true } },
    apiKey: () => OPENROUTER_KEY,
    supportsVision: false,
    contextLength: 128000,
    enabled: false, // 停用  ,慢且不稳定(日志采纳率 14% 垫底,p90=27.9s)
  },
  // G 代号曾用于 Gemini 3 Flash Preview,因 40% 失败率移除( 前),代号保留作历史标记,不复用
  {
    id: 'H', name: 'Gemini 3.1 Flash Lite Preview', shortName: 'Gem3.1L', provider: 'Google',
    baseUrl: OPENROUTER_BASE,
    model: 'google/gemini-3.1-flash-lite-preview',
    apiKey: () => OPENROUTER_KEY,
    supportsVision: true,
    contextLength: 1000000,
    extraBody: { reasoning: { max_tokens: 200 } },
  },
  {
    id: 'I', name: 'DeepSeek V4 Flash', shortName: 'DSV4', provider: 'DeepSeek',
    // I 从议会成员升级为 Chairman/Router(原因:中文榜接近 #1 + 30 tools schema 决策完美,
    // 替代 GPT-5.4 Nano 解决"大 schema 生成不完整 JSON"+"幻觉禁用成员"+"web_search 错判"系列问题)。
    // 代号 I 保留绑定 DSV4 Flash,但 enabled:false 不再当议会成员。Chairman 配置见 CHAIRMAN_PRIMARY。
    baseUrl: DEEPSEEK_BASE,
    model: 'deepseek-v4-flash',
    apiKey: () => DEEPSEEK_KEY,
    supportsVision: false,
    contextLength: 1000000,  // Note:V4 系列已升 1M,之前 128K 是早期版本数据
    extraBody: { thinking: { type: 'disabled' } },
    enabled: false, // 升 Chairman 后不再作议会成员
  },
  // J 新增 :GPT-5.4 Nano 从 Chairman 降级为议会成员。
  // 它当 Chairman 已知问题:幻觉禁用成员(MiMo:0)、大 tools schema 生成不完整 JSON、web_search 错判常见。
  // 但作议会成员仍能贡献:指令跟随稳、不输出 native markup、能力中等。replace 原 DSV4 在成员位置。
  {
    id: 'J', name: 'GPT-5.4 Nano', shortName: 'GPTN', provider: 'OpenAI',
    baseUrl: OPENROUTER_BASE,
    model: 'openai/gpt-5.4-nano',
    apiKey: () => OPENROUTER_KEY,
    supportsVision: true,
    contextLength: 400000,
    extraBody: { reasoning: { effort: 'low' } },
  },
];

// Chairman/Router 升级为 DeepSeek V4 Flash 直连。
// 原 GPT-5.4 Nano 在大 tools schema(~34 tools)下生成不完整 JSON 必失败,web_search 错判,
// MiMo:0 幻觉,any client requiring tool calling 的场景都坏。
// curl gate 实测 DSV4 Flash 在 30 tools schema 下 5/5 决策正确,1-2s 响应,JSON 完整。
// 必须 thinking:disabled — 多轮场景下 reasoning_content round-trip 会触发 400。
const CHAIRMAN_PRIMARY = {
  id: 'Chairman', name: 'DeepSeek V4 Flash', provider: 'DeepSeek',
  baseUrl: DEEPSEEK_BASE,
  model: 'deepseek-v4-flash',
  apiKey: () => DEEPSEEK_KEY,
  extraBody: { thinking: { type: 'disabled' } },
  contextLength: 1000000,  // Note:V4 Flash 是 1M,之前 128K 是早期版本数据
  maxOutputTokens: 8000,
};

const CHAIRMAN_FALLBACK = {
  id: 'Chairman-FB', name: 'Grok 4.1 Fast', provider: 'xAI',
  baseUrl: OPENROUTER_BASE,
  model: 'x-ai/grok-4.1-fast',
  apiKey: () => OPENROUTER_KEY,
  extraBody: { reasoning: { effort: 'low' } },
  contextLength: 2000000,
  maxOutputTokens: 8000,
};

// ─── Front-End Router (single-agent LLM, tool decision + dispatch) ───────

/**
 * Check if messages contain tool results (indicating a return trip from Gateway).
 */
function hasToolResults(messages) {
  return messages.some(m => m.role === 'tool' || m.role === 'function');
}

/**
 * Validate a tool_call against the tools schema provided by Gateway.
 * Returns { valid: true } or { valid: false, error: "..." }.
 */
function validateToolCall(toolCall, toolsSchema) {
  if (!toolCall?.function?.name) {
    return { valid: false, error: 'Missing function name' };
  }

  const funcName = toolCall.function.name;
  const toolDef = toolsSchema.find(t =>
    t.function?.name === funcName || t.name === funcName
  );

  if (!toolDef) {
    return { valid: false, error: `Unknown tool: ${funcName}. Available: ${toolsSchema.map(t => t.function?.name || t.name).join(', ')}` };
  }

  // Parse arguments
  let args;
  try {
    args = typeof toolCall.function.arguments === 'string'
      ? JSON.parse(toolCall.function.arguments)
      : toolCall.function.arguments || {};
  } catch (e) {
    return { valid: false, error: `Invalid JSON in arguments: ${e.message}` };
  }

  // Check required parameters from schema
  const params = toolDef.function?.parameters || toolDef.parameters || {};
  const required = params.required || [];
  for (const req of required) {
    if (!(req in args)) {
      return { valid: false, error: `Missing required parameter: ${req}` };
    }
  }

  return { valid: true };
}

// ─── DSV4 Protocol Adapter (, ) ──────────────────────────
// 集中处理 DSV4 跟标准 OpenAI Chat Completions 协议的差异。
// 只对 provider === 'DeepSeek' 的 config 应用,其他模型路径不受影响。
// 已适配陷阱清单:
//   #1 thinking 必须 disabled():由 CHAIRMAN_PRIMARY.extraBody 配置
//   #2 messages 中 image_url(及其他非 text)content 必须 strip()
//   #3 assistant tool_calls 后必须立即跟 role:tool message(frontEndRoute retry)
//   #4 timeout 至少 45s(CHAIRMAN_TIMEOUT_MS / ROUTER_TIMEOUT_MS)
//   #5 tool_calls JSON max_tokens 至少 4000(ROUTER_MAX_TOKENS)
//   #6 工具调用缺 required default-able 字段时自动补默认值(17:52 patch missing mode)
//   #7 messages 中 reasoning_content 在 thinking disabled 时必须 strip(19:45 fail)
//   #8 routeRequest 必须 spread routerConfig.extraBody 到 body(20:00 真正修)
// 第二层(出站参数净化):dsv4SanitizeBody —— strip n/user/seed 等议会用不到的字段
// 第三层(错误纠错重试):dsv4AttemptRecovery —— 按 error.message pattern 匹配修复 + 重发
//   pattern 1: "Missing reasoning_content field at message index N" → 补 N 的空字符串
//   pattern 2: "Messages with role 'tool' must be response" → 删孤儿 tool
//   pattern 3: "Content Exists Risk" → 不重试,标记需 fallback
//   pattern 4: "context length exceeded" → 截断历史

function isDeepSeekProvider(config) {
  return config?.provider === 'DeepSeek';
}

/**
 * Strip reasoning_content from assistant message history (DSV4 trap #7).
 * DSV4 with thinking:disabled rejects requests where messages contain
 * historical reasoning_content; must remove before sending.
 */
function dsv4StripReasoningContent(messages) {
  return messages.map(msg => {
    if (msg.role === 'assistant' && 'reasoning_content' in msg) {
      const { reasoning_content, ...rest } = msg;
      return rest;
    }
    return msg;
  });
}

/**
 * Strip image_url content type, keeping only text (DSV4 trap #2).
 * DSV4 returns HTTP 400 "unknown variant image_url" when receiving multimodal.
 */
function dsv4StripImageContent(messages) {
  return messages.map(msg => {
    if (Array.isArray(msg.content)) {
      const textParts = msg.content
        .filter(c => c.type === 'text')
        .map(c => c.text || '')
        .join(' ');
      return { ...msg, content: textParts || '[user attached an image]' };
    }
    return msg;
  });
}

/**
 * Apply all DSV4 message-level adaptations.
 * Returns a new messages array; original is not mutated.
 */
function dsv4PrepareMessages(messages, options = {}) {
  let prepared = messages;
  if (options.stripReasoning !== false) {
    prepared = dsv4StripReasoningContent(prepared);
  }
  if (options.stripMultimodal !== false) {
    prepared = dsv4StripImageContent(prepared);
  }
  return prepared;
}

/**
 * Enhanced version of validateToolCall: also auto-completes missing required
 * parameters when the schema provides a default value (DSV4 trap #6).
 *
 * Returns:
 *   { valid: true, toolCall, completed: [...] }  — toolCall args may be augmented
 *   { valid: false, error }                       — fundamentally invalid
 *
 * Use this instead of validateToolCall() for DSV4 paths.
 */
function dsv4ValidateAndCompleteToolCall(toolCall, toolsSchema) {
  if (!toolCall?.function?.name) {
    return { valid: false, error: 'Missing function name' };
  }

  const funcName = toolCall.function.name;
  const toolDef = toolsSchema.find(t =>
    t.function?.name === funcName || t.name === funcName
  );

  if (!toolDef) {
    return { valid: false, error: `Unknown tool: ${funcName}` };
  }

  let args;
  try {
    args = typeof toolCall.function.arguments === 'string'
      ? JSON.parse(toolCall.function.arguments)
      : toolCall.function.arguments || {};
  } catch (e) {
    return { valid: false, error: `Invalid JSON in arguments: ${e.message}` };
  }

  const params = toolDef.function?.parameters || toolDef.parameters || {};
  const required = params.required || [];
  const properties = params.properties || {};
  const completed = [];

  for (const req of required) {
    if (!(req in args)) {
      const propSchema = properties[req] || {};
      if ('default' in propSchema) {
        args[req] = propSchema.default;
        completed.push(`${req}=${JSON.stringify(propSchema.default)}`);
        continue;
      }
      return { valid: false, error: `Missing required parameter: ${req} (no schema default)` };
    }
  }

  if (completed.length > 0) {
    console.log(`  🔧 DSV4 adapter: auto-completed ${funcName}: ${completed.join(', ')}`);
    return {
      valid: true,
      toolCall: {
        ...toolCall,
        function: {
          ...toolCall.function,
          arguments: JSON.stringify(args),
        },
      },
      completed,
    };
  }

  return { valid: true, toolCall, completed: [] };
}

/**
 * Inject a short Router guidance preamble to the first system message,
 * encouraging the Router to actively call honcho_* tools when appropriate.
 * 通过 prompt 工程让议会调度员更灵敏使用 5 个 honcho 工具,
 * 替代昂贵的"议会成员暴露 Honcho 工具"架构改造。
 *
 * Idempotent: 如果 system 里已含同样标记,不重复注入。
 */
const ROUTER_HONCHO_GUIDANCE = `

[议会调度提示]
你看到的 honcho_search / honcho_profile / honcho_conclude / honcho_context / honcho_reasoning 5 个工具是主动记忆调用。
- 用户提"我之前/上次/历史/我的偏好/我对X的看法" → 调 honcho_search 或 honcho_profile
- 用户明确说"记一下/记住/归档" → 调 honcho_conclude
- 复杂推理需要综合历史上下文 → 偶尔调 honcho_reasoning(贵,慎用)
- 普通对话(system 已含 memory-context)→ 走议会共识,不调 honcho

[严格规则]
你只能调用 tools schema 里实际提供的工具。不要发明、推测或假设列表外的工具存在。
如果回答需要某个工具但它不在 schema 里,直接走议会共识让成员凭已有信息回答,不要幻觉调用。`;

function injectRouterGuidance(messages) {
  const marker = '[议会调度提示]';
  const firstSystemIdx = messages.findIndex(m => m.role === 'system');
  if (firstSystemIdx === -1) {
    return [{ role: 'system', content: ROUTER_HONCHO_GUIDANCE.trim() }, ...messages];
  }
  const sys = messages[firstSystemIdx];
  const sysContent = typeof sys.content === 'string' ? sys.content : '';
  if (sysContent.includes(marker)) return messages; // 已注入过,不重复
  const enhanced = [...messages];
  enhanced[firstSystemIdx] = { ...sys, content: sysContent + ROUTER_HONCHO_GUIDANCE };
  return enhanced;
}

// ─── 搜索后端状态机 (, ) ──────────────────────────────────
// 搭配 /opt/council-proxy/scripts/search_state_manager.py 的状态机,
// 议会代理读 .search_state.json 决定 SEARXNG_ONLY 状态下是否主动调 SearXNG。
// 状态:
//   TAVILY        host uses Tavily 主搜索,议会代理 SearXNG 仅做 fallback 补充
//   EXA           host uses Exa 备搜索,议会代理 SearXNG 仅做 fallback 补充
//   SEARXNG_ONLY  host does not enable web toolset,议会代理在用户搜索意图触发时主动调 SearXNG

const SEARCH_STATE_PATH = process.env.SEARCH_STATE_PATH ||
  join(process.env.HOME || '/tmp', '.council-proxy', 'search_state.json');

const SEARCH_INTENT_KEYWORDS = [
  // 中文
  '查', '搜', '搜索', '今天', '最新', '现在', '当前', '查询',
  '天气', '气温', '价格', '汇率', '股价', '新闻', '政策', '消息',
  // 英文
  'today', 'latest', 'current', 'weather', 'price', 'news', 'search',
];

function readSearchState() {
  try {
    const data = JSON.parse(readFileSync(SEARCH_STATE_PATH, 'utf8'));
    return data.state || 'TAVILY';
  } catch {
    return 'TAVILY'; // 文件不存在或损坏 → 默认 TAVILY 状态(原行为)
  }
}

function containsSearchIntent(text) {
  if (!text || typeof text !== 'string' || text.length < 4) return false;
  const lower = text.toLowerCase();
  return SEARCH_INTENT_KEYWORDS.some(k => lower.includes(k.toLowerCase()));
}

function extractSearchQuery(userMsg) {
  if (!userMsg || typeof userMsg !== 'string') return '';
  return userMsg.slice(0, 200).trim();
}

/**
 * 调用本地 SearXNG 拿前 N 个搜索结果,返回格式化字符串(议会成员能直接读)。
 * 失败/超时返回 null。
 */
async function callSearxng(query, limit = 5) {
  if (!query) return null;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 3000);
  try {
    const res = await fetch(
      `http://127.0.0.1:8888/search?q=${encodeURIComponent(query)}&format=json&categories=general`,
      { signal: ctrl.signal }
    );
    clearTimeout(timer);
    const data = await res.json();
    const results = (data.results || []).slice(0, limit);
    if (results.length === 0) return null;
    return results
      .map((r, i) => `${i + 1}. ${r.title || '?'} - ${r.url || ''}\n   ${(r.content || '').slice(0, 150)}`)
      .join('\n');
  } catch (err) {
    console.log(`  [~] SearXNG: skipped (${(err.message || '').slice(0, 50)})`);
    return null;
  }
}

/**
 * Strip request body fields that DSV4 doesn't support or that might
 * cause confusion in logs (n/user/seed). Returns a new body object.
 *  第二层加固。
 */
function dsv4SanitizeBody(body) {
  if (!body || typeof body !== 'object') return body;
  const { n, user, seed, ...rest } = body;
  return rest;
}

/**
 *  第三层:错误纠错重试 dispatcher。
 * 解析 DSV4 返回的 HTTP 错误,匹配已知 pattern,返回修复后的 messages。
 *
 * @param errorMessage  DSV4 返回的 error.message 字符串
 * @param messages      原始发出的 messages 数组
 * @returns {
 *   recovered: boolean,
 *   strategy: string,    // 匹配到的 pattern 名(用于日志)
 *   messages?: array,    // 修复后的 messages(recovered=true 时返回)
 *   noRetry?: boolean,   // 不应重试(如内容审核),需走 fallback
 * }
 */
function dsv4AttemptRecovery(errorMessage, messages) {
  if (!errorMessage || !Array.isArray(messages)) {
    return { recovered: false, strategy: 'invalid-input' };
  }

  // Pattern 1: "Missing reasoning_content field in the assistant message at message index N"
  const missingIdxMatch = errorMessage.match(/Missing.*reasoning_content.*at message index (\d+)/i);
  if (missingIdxMatch) {
    const idx = parseInt(missingIdxMatch[1], 10);
    if (idx >= 0 && idx < messages.length && messages[idx].role === 'assistant') {
      const fixed = [...messages];
      fixed[idx] = { ...fixed[idx], reasoning_content: '' };
      console.log(`  🔧 DSV4 recovery: pattern=missing-reasoning-idx, filled message[${idx}].reasoning_content=""`);
      return { recovered: true, strategy: 'missing-reasoning-idx', messages: fixed };
    }
  }

  // Pattern 1b: "The reasoning_content in the thinking mode must be passed back"
  // (This is the bare-form variant; we already strip on outbound but if it still slips, strip more aggressively.)
  if (/reasoning_content.*thinking mode.*passed back/i.test(errorMessage)) {
    const fixed = dsv4StripReasoningContent(messages);
    console.log('  🔧 DSV4 recovery: pattern=reasoning-passback, force-stripped all reasoning_content');
    return { recovered: true, strategy: 'reasoning-passback', messages: fixed };
  }

  // Pattern 2: "Messages with role 'tool' must be a response to a preceding message with 'tool_calls'"
  if (/role.*['"]tool['"].*response.*tool_calls/i.test(errorMessage)) {
    // 删除任何前面不是 assistant+tool_calls 的孤儿 tool message
    const fixed = [];
    for (let i = 0; i < messages.length; i++) {
      const msg = messages[i];
      if (msg.role === 'tool') {
        const prev = messages[i - 1];
        const prevHasToolCalls = prev?.role === 'assistant' && Array.isArray(prev.tool_calls) && prev.tool_calls.length > 0;
        if (!prevHasToolCalls) {
          console.log(`  🔧 DSV4 recovery: pattern=orphan-tool, dropped messages[${i}] (orphan tool)`);
          continue;
        }
      }
      fixed.push(msg);
    }
    if (fixed.length !== messages.length) {
      return { recovered: true, strategy: 'orphan-tool', messages: fixed };
    }
  }

  // Pattern 3: "Content Exists Risk" — 内容审核,不重试,需要 fallback
  if (/Content Exists Risk/i.test(errorMessage)) {
    console.log('  🚨 DSV4 recovery: pattern=content-risk, noRetry → fallback');
    return { recovered: false, strategy: 'content-risk', noRetry: true };
  }

  // Pattern 4: "context length exceeded" / 类似措辞
  if (/context.{0,20}(length|window).{0,20}(exceed|too long)/i.test(errorMessage)) {
    // 简单截断:保留 system + 最后 8 条 message
    const systemMsgs = messages.filter(m => m.role === 'system');
    const nonSystem = messages.filter(m => m.role !== 'system');
    if (nonSystem.length > 8) {
      const fixed = [...systemMsgs, ...nonSystem.slice(-8)];
      console.log(`  🔧 DSV4 recovery: pattern=context-overflow, truncated ${messages.length} → ${fixed.length}`);
      return { recovered: true, strategy: 'context-overflow', messages: fixed };
    }
  }

  // 未匹配到 pattern — log 给以后扩 checklist
  const preview = errorMessage.slice(0, 150).replace(/\n/g, ' ');
  console.log(`  ❓ DSV4 recovery: no pattern matched for: ${preview}`);
  return { recovered: false, strategy: 'unknown' };
}

// ─── End DSV4 Protocol Adapter ───────────────────────────────────────────────

/**
 * Ask Router (single-agent LLM) whether this request needs tool calling.
 * Returns: { needsTool: false } or { needsTool: true, toolCalls: [...] }
 */
async function routeRequest(messages, toolsSchema) {
  const routerConfig = {
    ...CHAIRMAN_PRIMARY,
    id: 'Router',
    fixedTemperature: 0.3,  // low temperature for deterministic routing
    // bug fix:不再覆盖 extraBody,继承 CHAIRMAN_PRIMARY 的 thinking:disabled。
    // 原 extraBody:{reasoning:{effort:medium}} 是给 GPT-5.4 Nano 调的,DSV4 关思考时直接覆盖
    // 会让 DSV4 默认开思考,多轮场景触发 reasoning_content round-trip 400 错误。
  };

  // 改用统一 DSV4 适配层(traps #2 + #7)。
  // 原 inline stripMultimodal 已迁到 dsv4StripImageContent,reasoning_content strip 是新增。
  // 议会成员调用 callModel 时也会按 isDeepSeekProvider 自动判断是否走适配。
  //  后续:在 Router 路径注入 honcho 工具引导,让议会调度员更灵敏使用 5 个记忆工具。
  // 这条引导只影响 Router(routeRequest 内部),不影响议会成员看到的 messages。
  const adaptedMessages = isDeepSeekProvider(routerConfig)
    ? dsv4PrepareMessages(messages)
    : messages;
  const routerMessages = injectRouterGuidance(adaptedMessages);

  // Call Router with tools schema so it can generate tool_calls natively
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ROUTER_TIMEOUT_MS);

  try {
    const response = await fetch(`${routerConfig.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${routerConfig.apiKey()}`,
      },
      body: JSON.stringify(
        //  第二层:DSV4 出站参数净化(strip n/user/seed 等议会用不到的字段)
        // 关键 bug fix —  改了 routerConfig 继承 extraBody,
        // 但 routeRequest 这里 fetch body 没 spread extraBody,导致 thinking:disabled 实际没传给 DSV4。
        // 结果:thinking 默认 enabled,messages 历史里的 reasoning_content 触发 DSV4 严格校验 400。
        (isDeepSeekProvider(routerConfig) ? dsv4SanitizeBody : (x => x))({
          model: routerConfig.model,
          messages: routerMessages,
          tools: toolsSchema,
          tool_choice: 'auto',
          max_tokens: ROUTER_MAX_TOKENS,
          temperature: 0.3,
          stream: false,
          ...(routerConfig.extraBody || {}),
        })
      ),
      signal: controller.signal,
    });

    if (!response.ok) {
      const errText = await response.text().catch(() => 'unknown');
      throw new Error(`Router HTTP ${response.status}: ${errText.slice(0, 200)}`);
    }

    const data = await response.json();
    const choice = data.choices?.[0];

    if (!choice) {
      throw new Error('Router returned empty choices');
    }

    // Check if model decided to call a tool
    if (choice.message?.tool_calls?.length > 0) {
      return { needsTool: true, toolCalls: choice.message.tool_calls, usage: data.usage };
    }

    // Model decided no tool needed
    return { needsTool: false, usage: data.usage };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Front-end routing with validation and retry.
 * Returns:
 *   { action: 'consensus' }           → go to council consensus flow
 *   { action: 'tool_call', response } → return tool_call to Gateway
 *   { action: 'error', message }      → return error to user
 */
async function frontEndRoute(messages, toolsSchema) {
  // No tools provided by Gateway → skip routing, go to consensus
  if (!toolsSchema || toolsSchema.length === 0) {
    console.log('  🔀 Router: no tools from Gateway, skipping → consensus');
    return { action: 'consensus' };
  }

  // 检测用户引用 #NNN 议会编号,加载 archive 注入到 user message
  // 让议会成员 + Chairman 都能看到过往议会原文(绕过 client's context window compression)
  // 例如用户说"对比 #015 跟 #042 的差异"或"#042 那次 Gem3.1L 异议是什么",议会代理读 archive 注入
  //  ( 修正):原版用 push system message,但 dispatchToMembers 过滤所有 system,
  // Chairman 也不看 system,导致注入失效。改成 mutate 最后一条 user message content 直接 append,
  // 这样议会成员看到的 user msg + Chairman 的 originalQuestion 都包含 archive。
  const lastUser = [...messages].reverse().find(m => m.role === 'user');
  const lastUserContent = typeof lastUser?.content === 'string' ? lastUser.content : '';
  const turnRefs = [...new Set([...lastUserContent.matchAll(/#(\d{1,6})/g)].map(m => parseInt(m[1], 10)))];
  if (turnRefs.length > 0 && turnRefs.length <= 5 && lastUser && typeof lastUser.content === 'string') {
    const archives = turnRefs.map(id => readCouncilArchive(id)).filter(Boolean);
    if (archives.length > 0) {
      const injection = buildArchiveInjectionMessage(archives);
      // 把 archive append 到 user message 末尾 — 议会成员看 user msg 时看到 + Chairman 通过 originalQuestion 也看到
      lastUser.content = lastUser.content + '\n\n' + injection;
      console.log(`  📦 Archive injected to user msg: ${archives.length} turn(s) → #${archives.map(a => String(a.id).padStart(3, '0')).join(', #')}`);
    }
  }

  // 搜索后端状态机 — SEARXNG_ONLY 状态时跳过 Router,议会代理主动调 SearXNG
  // 触发条件:state=SEARXNG_ONLY + 这是用户新轮次(无 tool result 历史) + 消息含搜索意图关键词
  // 目的:Tavily + Exa 都 quota 用尽时仍能给议会成员注入实时数据
  const searchState = readSearchState();
  if (searchState === 'SEARXNG_ONLY' && !hasToolResults(messages)) {
    const lastUser = [...messages].reverse().find(m => m.role === 'user');
    const userText = typeof lastUser?.content === 'string' ? lastUser.content : '';
    if (containsSearchIntent(userText)) {
      const query = extractSearchQuery(userText);
      console.log(`  🔍 [SEARXNG_ONLY] 主动 SearXNG 搜索: "${query.slice(0, 60)}"`);
      const results = await callSearxng(query);
      if (results) {
        messages.push({
          role: 'system',
          content: `[SearXNG 主动搜索结果 (主备搜索均用尽,补搜模式) for "${query.slice(0, 60)}"]\n${results}`,
        });
        console.log(`  [+] SearXNG 主动注入,议会成员可见结果`);
      } else {
        console.log(`  [~] SearXNG 无结果或失败,议会成员将走知识边界引导`);
      }
    }
    return { action: 'consensus' };
  }

  // Multi-step tool calling: per-question counting (resets on each new user message)
  if (hasToolResults(messages)) {
    const lastUserIdx = messages.reduce((last, m, i) => m.role === 'user' ? i : last, -1);
    const currentQMsgs = messages.slice(lastUserIdx + 1);
    const toolCallRounds = currentQMsgs.filter(m =>
      m.role === 'assistant' && m.tool_calls?.length > 0
    ).length;

    // Detect stuck loop: 5 consecutive calls to the same tool
    const recentToolNames = currentQMsgs
      .filter(m => m.role === 'assistant' && m.tool_calls?.length > 0)
      .slice(-5)
      .map(m => m.tool_calls[0]?.function?.name);
    const stuckInLoop = recentToolNames.length >= 5 && recentToolNames.every(n => n === recentToolNames[0]);

    if (stuckInLoop) {
      console.log('  [!] Router: stuck loop (' + recentToolNames[0] + ' x5), pausing');
      return {
        action: 'limit_reached',
        message: 'Detected repeated tool calls (' + recentToolNames[0] + ' x5). Paused to avoid loop. Reply to continue or rephrase.',
      };
    }

    // 升级版 — 检测 search-class 跨 tool 死循环
    // 场景:Alfred 反复换搜索 tool(honcho_search → session_search → web_search 等)但都查不到内容,
    // 跟"同 tool 5 次"的 stuckInLoop 不同,这是"换着 tool 都搜不到结果"。直接切 consensus 让议会成员凭已有信息答。
    const SEARCH_CLASS_TOOLS = ['honcho_search', 'session_search', 'web_search', 'search_files', 'honcho_context'];
    const allSearchClass = recentToolNames.length >= 5 && recentToolNames.every(n => SEARCH_CLASS_TOOLS.includes(n));
    if (allSearchClass && !stuckInLoop) {
      console.log(`  [!] Router: 5 consecutive search-class calls (${recentToolNames.join(', ')}) — forcing consensus`);
      return { action: 'consensus' };
    }

    if (toolCallRounds >= TOOL_CALL_MAX_ROUNDS) {
      const completedTools = currentQMsgs
        .filter(m => m.role === 'tool')
        .map(m => m.name || 'tool')
        .slice(-5);
      console.log(`  Router: max tool rounds (${TOOL_CALL_MAX_ROUNDS}) reached for this question`);
      return {
        action: 'limit_reached',
        message: `Executed ${toolCallRounds} tool rounds for this question (${completedTools.join(' > ')}). Reply to continue or rephrase.`,
      };
    }
    console.log(`  Router: tool results detected (round ${toolCallRounds}/${TOOL_CALL_MAX_ROUNDS}), checking if more tools needed...`);

    // --- SearXNG supplementary search ---
    const lastAWT = [...messages].reverse()
      .find(m => m.role === 'assistant' && m.tool_calls?.length > 0);
    const wsc = lastAWT?.tool_calls?.find(tc => tc.function?.name === 'web_search');
    if (wsc) {
      try {
        const wsArgs = typeof wsc.function.arguments === 'string'
          ? JSON.parse(wsc.function.arguments) : wsc.function.arguments || {};
        const wsQ = wsArgs.query || wsArgs.q || '';
        if (wsQ) {
          const sCtrl = new AbortController();
          const sTimer = setTimeout(() => sCtrl.abort(), 3000);
          const sRes = await fetch(
            `http://127.0.0.1:8888/search?q=${encodeURIComponent(wsQ)}&format=json&categories=general`,
            { signal: sCtrl.signal }
          );
          clearTimeout(sTimer);
          const sData = await sRes.json();
          const sResults = (sData.results || []).slice(0, 5);
          if (sResults.length > 0) {
            const supp = sResults
              .map((r, i) => `${i+1}. ${r.title || '?'} - ${r.url || ''}\n   ${(r.content || '').slice(0, 150)}`)
              .join('\n');
            const lastWR = [...messages].reverse()
              .find(m => m.role === 'tool' && (m.name === 'web_search' || m.name === 'search'));
            const pFailed = lastWR?.content && (
              lastWR.content.includes('503') || lastWR.content.includes('error') || lastWR.content.length < 50);
            messages.push({
              role: 'system',
              content: `[SearXNG supplementary results for "${wsQ}"]\n${supp}`,
            });
            if (pFailed) {
              console.log('  [!] Primary search failed - SearXNG providing ' + sResults.length + ' fallback results');
            } else {
              console.log('  [+] SearXNG: appended ' + sResults.length + ' supplementary results');
            }
          }
        }
      } catch (sErr) {
        console.log('  [~] SearXNG: skipped (' + (sErr.message || '').slice(0, 50) + ')');
      }
    }

    // Fall through to let Router decide
  }

  console.log(`  🔀 Router: Router analyzing request (${toolsSchema.length} tools available)...`);

  //  第三层:recovery 可能修过 messages,后续 attempt + fallback 都用 currentMessages
  let currentMessages = messages;

  for (let attempt = 1; attempt <= ROUTER_MAX_RETRIES; attempt++) {
    try {
      const result = await routeRequest(currentMessages, toolsSchema);

      if (!result.needsTool) {
        const tokens = result.usage ? `${result.usage.prompt_tokens || '?'}in/${result.usage.completion_tokens || '?'}out` : '';
        console.log(`  🔀 Router: no tool needed (${tokens}) → consensus`);
        return { action: 'consensus' };
      }

      // 用 DSV4 适配层的 validateAndComplete 替代 validateToolCall,
      // 自动补 default-able 字段(trap #6)。补完的 toolCall 用于后续 SSE 响应。
      let allValid = true;
      const errors = [];
      const completedToolCalls = [];
      for (const tc of result.toolCalls) {
        const validation = dsv4ValidateAndCompleteToolCall(tc, toolsSchema);
        if (!validation.valid) {
          allValid = false;
          errors.push(validation.error);
        } else {
          completedToolCalls.push(validation.toolCall);
        }
      }

      if (allValid) {
        const toolNames = completedToolCalls.map(tc => tc.function?.name).join(', ');
        console.log(`  🔀 Router: tool call validated ✅ → ${toolNames} (attempt ${attempt})`);
        return {
          action: 'tool_call',
          response: {
            id: `council-${Date.now()}`,
            object: 'chat.completion',
            created: Math.floor(Date.now() / 1000),
            model: 'council-v1/router',
            choices: [{
              index: 0,
              message: {
                role: 'assistant',
                content: null,
                tool_calls: completedToolCalls,
              },
              finish_reason: 'tool_calls',
            }],
            usage: result.usage || {},
          },
        };
      }

      // Validation failed
      console.log(`  🔀 Router: validation failed (attempt ${attempt}/${ROUTER_MAX_RETRIES}): ${errors.join('; ')}`);

      // 兜底 — 如果所有验证失败都是 "Unknown tool"(LLM 幻觉调用不存在工具),
      // 不浪费 retry 也不返回错误给用户,直接走 consensus 让议会成员凭已有信息答 + 必要时明确说"无法获取 X"。
      // 通用机制:不点名具体工具,任何不在 schema 的工具调用都自动 fallback。
      const allUnknownTool = errors.length > 0 && errors.every(e => e.includes('Unknown tool:'));
      if (allUnknownTool) {
        console.log(`  ⚠️ Router: all errors are "Unknown tool" (LLM hallucination) → fallback to consensus`);
        return { action: 'consensus' };
      }

      if (attempt < ROUTER_MAX_RETRIES) {
        // 不再 push (assistant+tool_calls + user feedback) 到 messages。
        // 这是 GPT-5.4 Nano 时代的 hack,DeepSeek 严格协议要求 assistant tool_calls 后必须跟
        // role:tool 消息,跟 user 立即触发 HTTP 400 "must be followed by tool"。
        // 改为 retry 用相同 messages,让 DSV4 自己重新生成(不同 random,可能成功)。
        // 主要靠 ROUTER_MAX_TOKENS 4000 防止再被截断。
      }
    } catch (err) {
      const reason = err.name === 'AbortError' ? 'timeout' : err.message?.slice(0, 100);
      console.log(`  🔀 Router: error (attempt ${attempt}/${ROUTER_MAX_RETRIES}): ${reason}`);

      //  第三层:错误纠错重试 — 解析 DSV4 错误信息,匹配已知 pattern,
      // 修复 messages 然后下一次 attempt 自动用新 messages 重发。
      // noRetry(如内容审核 Content Exists Risk)直接 break,让 caller 走 fallback 路径。
      try {
        let parsedErrMsg = err.message || '';
        // Router HTTP 400 错误 message 形如 "Router HTTP 400: {\"error\":{\"message\":\"...\"}}"
        // 尝试解析 JSON,只有 inner message 才是 DSV4 真正的错误描述
        const httpMatch = parsedErrMsg.match(/Router HTTP \d+:\s*(\{.+\})/);
        if (httpMatch) {
          try {
            const errJson = JSON.parse(httpMatch[1]);
            if (errJson?.error?.message) parsedErrMsg = errJson.error.message;
          } catch {}
        }
        const recovery = dsv4AttemptRecovery(parsedErrMsg, currentMessages);
        if (recovery.recovered) {
          currentMessages = recovery.messages;
          console.log(`  🔁 Router: applied recovery=${recovery.strategy}, retrying with fixed messages`);
        } else if (recovery.noRetry) {
          console.log(`  🚨 Router: ${recovery.strategy} — no retry, breaking out`);
          break;
        }
      } catch (recErr) {
        console.log(`  ⚠️ Router: recovery itself errored: ${recErr.message?.slice(0, 80)}`);
      }

      if (attempt >= ROUTER_MAX_RETRIES) {
        // Try router retry (same model)
        if (attempt === ROUTER_MAX_RETRIES) {
          console.log('  🔀 Router: retrying...');
          try {
            const fallbackResult = await routeRequest(currentMessages, toolsSchema);
            if (!fallbackResult.needsTool) {
              console.log('  🔀 Router (retry): no tool needed → consensus');
              return { action: 'consensus' };
            }
            // 用 DSV4 适配层 validateAndComplete(同上 trap #6)。
            let retryValid = true;
            const retryCompletedToolCalls = [];
            for (const tc of fallbackResult.toolCalls) {
              const validation = dsv4ValidateAndCompleteToolCall(tc, toolsSchema);
              if (!validation.valid) {
                retryValid = false;
              } else {
                retryCompletedToolCalls.push(validation.toolCall);
              }
            }
            if (retryValid) {
              console.log('  🔀 Router (retry): tool call validated ✅');
              return {
                action: 'tool_call',
                response: {
                  id: `council-${Date.now()}`,
                  object: 'chat.completion',
                  created: Math.floor(Date.now() / 1000),
                  model: 'council-v1/router-fallback',
                  choices: [{
                    index: 0,
                    message: { role: 'assistant', content: null, tool_calls: retryCompletedToolCalls },
                    finish_reason: 'tool_calls',
                  }],
                  usage: fallbackResult.usage || {},
                },
              };
            }
          } catch (fbErr) {
            console.log(`  🔀 Router (retry): also failed: ${fbErr.message?.slice(0, 80)}`);
          }
        }
      }
    }
  }

  // All retries exhausted
  console.log('  🔀 Router: all retries failed → returning error');
  return { action: 'error', message: '⚠️ 工具调用失败，请稍后重试或换一种方式提问。' };
}

// ─── System Prompt Processing ───────────────────────────────────────────────

/**
 * 从 client-provided system prompt 中抽取给议会成员的紧凑上下文摘要。
 * host application's ephemeral channel_prompt 通常 append 在 system 尾部,这里抽 "最后一段" +
 * 用户称呼 hint。控制在约 500 字符,避免成员 input token × 6 倍爆炸。
 *  新增()。
 */
function extractMemberContextBrief(originalSystemPrompt) {
  if (!originalSystemPrompt || originalSystemPrompt.length === 0) return '';

  const parts = [];

  // 1. 用户称呼:找 Arthur 出现,给成员稳定的称呼锚点
  if (/Arthur/.test(originalSystemPrompt)) {
    parts.push('用户:Arthur(用自然、亲切的方式称呼)');
  }

  // 2. 环境/频道上下文:取 system prompt 尾部 ~500 字符
  //    host appends channel_prompt 作为 ephemeral append 在尾部,大概率包含频道规则和会话元数据
  const tailLen = Math.min(500, originalSystemPrompt.length);
  const tail = originalSystemPrompt.slice(-tailLen).trim();
  if (tail.length > 0) {
    const hasChannelMarker = /#\S+|你现在在|当前频道|channel/i.test(tail);
    const label = hasChannelMarker ? '当前环境/频道规则' : '当前环境上下文(系统 prompt 尾部片段)';
    parts.push(`${label}:\n${tail}`);
  }

  if (parts.length === 0) return '';
  return '\n\n---\n\n## 当前上下文(议会成员参考,不要在回答中暴露议会机制)\n\n' + parts.join('\n\n');
}

/**
 * 构造议会成员的 system prompt。
 *  修正():成员不再完全剥离 user context。在泛型 preamble 基础上,
 * 追加从 client system 中抽取的"频道 + 用户称呼"摘要,约 500 字增量,
 * 让成员能认知环境但不会导致 token × 6 爆炸。
 */
// ─── Council Archive (, ) ─────────────────────────────────
// 议会成员答案完整原文持久化层。每次议会调用产出递增编号 #NNN,Chairman 输出
// 评估行带编号,Arthur 可在 Discord 引用编号(例如 "#042 那次 Gem3.1L 怎么说")
// 让 Chairman 取出对应议会回合的成员原文 + Chairman 输出,做对比/复述/审视。
// VPS 本地存储零成本,永久保留(不压缩不清理)。
// 议会代理用 root 跑(systemd),ARCHIVE_DIR 跟 secrets.env 同目录方便管理。

const ARCHIVE_DIR = process.env.ARCHIVE_DIR ||
  join(process.env.HOME || '/tmp', '.council-proxy', 'archive');
const COUNTER_FILE = `${ARCHIVE_DIR}/.counter`;

function ensureArchiveDir() {
  try { mkdirSync(ARCHIVE_DIR, { recursive: true }); } catch {}
}

function getNextCouncilTurnId() {
  ensureArchiveDir();
  let n = 0;
  try {
    n = parseInt(readFileSync(COUNTER_FILE, 'utf8').trim(), 10) || 0;
  } catch {}
  n += 1;
  try {
    writeFileSync(COUNTER_FILE, String(n));
  } catch (e) {
    console.log(`  ⚠️ Council archive: failed to write counter: ${e.message?.slice(0, 80)}`);
  }
  return n;
}

function writeCouncilArchive(turnId, originalQuestion, memberResponses, skippedMembers, chairmanContent, chairmanName) {
  ensureArchiveDir();
  const filename = `${ARCHIVE_DIR}/${String(turnId).padStart(6, '0')}.json`;
  const archive = {
    id: turnId,
    timestamp: new Date().toISOString(),
    user_msg: typeof originalQuestion === 'string' ? originalQuestion : JSON.stringify(originalQuestion),
    user_msg_excerpt: (typeof originalQuestion === 'string' ? originalQuestion : '').slice(0, 200),
    member_responses: memberResponses.map(r => ({
      id: r.id,
      shortName: r.shortName || r.name,
      name: r.name,
      content: r.content,
      elapsed: r.elapsed,
      usage: r.usage,
    })),
    //  ( 修正):存 skipped 成员 + 失败原因(原版漏存,导致 Chairman 不知道
    // 为何议会少人,容易瞎编"路由机制按问题类型选模型"等不存在的解释)
    skipped_members: (skippedMembers || []).map(s => ({
      id: s.id,
      shortName: s.shortName || s.name,
      name: s.name,
      reason: s.reason,
    })),
    chairman_output: chairmanContent,
    chairman_model: chairmanName,
  };
  try {
    writeFileSync(filename, JSON.stringify(archive, null, 2));
    console.log(`  📦 Archive saved: #${String(turnId).padStart(3, '0')}`);
  } catch (e) {
    console.log(`  ⚠️ Council archive: failed to save #${turnId}: ${e.message?.slice(0, 80)}`);
  }
}

function readCouncilArchive(turnId) {
  const filename = `${ARCHIVE_DIR}/${String(turnId).padStart(6, '0')}.json`;
  if (!existsSync(filename)) return null;
  try {
    return JSON.parse(readFileSync(filename, 'utf8'));
  } catch (e) {
    console.log(`  ⚠️ Council archive: failed to read #${turnId}: ${e.message?.slice(0, 80)}`);
    return null;
  }
}

function buildArchiveInjectionMessage(archives) {
  const blocks = archives.map(a => {
    const memberBlock = a.member_responses
      .map(r => `### ${r.shortName} 完整回答\n${r.content}`)
      .join('\n\n---\n\n');

    //  ( 修正):展示 skipped 成员 + 真实失败原因。
    // 让 Chairman 在被追问"为什么 Qwen 没回答"时能给真原因(HTTP 429 / timeout / 等),
    // 不再编造"路由机制按问题类型分配"这种不存在的解释。
    const skipped = a.skipped_members || [];
    const skippedBlock = skipped.length > 0
      ? `\n\n### 当时缺席/失败成员(议会 ${a.member_responses.length}/${a.member_responses.length + skipped.length})\n${skipped.map(s => `- **${s.shortName}**: ${s.reason}`).join('\n')}\n\n*(议会架构是 4 路并发调用,缺席通常是 API 失败 / rate limit / timeout 等技术原因,**不是**议会"按问题类型选模型"或"动态分配"。这种机制在我们议会架构里不存在。)*`
      : '';

    return `[议会回合 #${String(a.id).padStart(3, '0')} 成员原文 archive (${a.timestamp})]
用户原始问题: ${a.user_msg_excerpt}

${memberBlock}${skippedBlock}

### Chairman 当时综合答案
${a.chairman_output}`;
  });
  return `[council-internal: 以下是用户引用的过往议会回合的完整 archive,Chairman 可基于这些原文回答用户的追问/对比/审视类问题。**只在用户明确引用 #NNN 时使用,否则忽略**。]

${blocks.join('\n\n========\n\n')}`;
}

// ─── End Council Archive ────────────────────────────────────────────────────

function buildMemberSystemPrompt(originalSystemPrompt) {
  const memberPreamble = `You are a Council member — one of several AI models independently answering the same question. Your response will be evaluated alongside other models' responses by a Chairman model who will fuse the best answer.

Instructions:
- Answer the question directly and thoroughly
- Respond in the same language as the user (Chinese if the user writes in Chinese)
- Be concise but complete
- If you're uncertain about something, say so
- Do not mention that you are part of a Council or that your answer will be evaluated
- You do NOT have access to any tools (no web search, no file read/write, no code execution). Do not output <tool_call>, <｜DSML｜tool_calls>, <｜DSML｜invoke ...>, OpenAI-style tool_calls JSON, XML-style invocations, or ANY tool invocation markup whatsoever — including DeepSeek native markup. You MUST respond with ONLY natural language text. If you find yourself wanting to use a tool, instead describe in words what you would do. If the conversation context already contains search results or file contents from previous steps, use that information directly to answer.
- If you cannot answer due to insufficient information in the context, say so plainly — do not pretend to call tools or claim API errors.
- 知识边界(重要):If the user asks for any DYNAMIC / REAL-TIME information (weather, current prices, today's news, latest policy/regulation, real-time stock quotes, live event status, etc.) and that data is NOT already in the conversation context, you MUST plainly tell the user something like "我无法获取实时的 X 数据,建议你查询 Y"(or in English equivalent if user writes in English). Do NOT guess based on training data — your training has a cutoff date and the user knows it. Static general knowledge (geography, history, well-established medical/financial principles, etc.) is fine to answer from training.
- **具体时间/日期/约定/数字**(防幻觉, 加):任何具体的时间点、日期、数字、具体约定细节(例如"下次复盘节点是 X 月 Y 日"、"我们 N 周前讨论过 Z"、"上次议会 #042 决定了 W"这种内容),如果**不在当前 messages 历史里明确出现过**,**绝不要编造**。直接说"我不确定具体[时间/日期/数字]"或"我看不到该约定的历史记录"。client conversation history会被压缩,你看不到全部上下文是预期 — 找不到就承认,**不要凭推理"补全"细节**。
`;

  const contextBrief = extractMemberContextBrief(originalSystemPrompt);
  return memberPreamble + contextBrief;
}

/**
 * 构造议会 Chairman 的"融合任务" user message。
 *  重构():Chairman 的 system = client-provided system(不 slice,保持 Alfred 身份、
 * Memory、频道规则完整),融合指令 + 成员答案作为 user message。
 * 旧版 buildChairmanPrompt 把"slice 过的 identity + 任务"拼成一个 system 字符串已废弃,
 * 因为 slice(0, 6000) 会把 host appends 在 system 尾部的 channel_prompt 切掉,
 * 导致 Chairman 不认频道规则( 诊断 bug)。
 */
function buildFusionUserMessage(originalQuestion, memberResponses, skippedMembers, turnId = null) {
  const responsesBlock = memberResponses
    .map(r => `### 成员 ${r.shortName || r.name}\n${r.content}`)
    .join('\n\n---\n\n');  // Note:把"本次因技术原因失败"的和"代号固化禁用"的成员区分开。
  // 只把前者计入 totalCount(议会评估行的"参与: N/M"分母),禁用成员不该算进去,
  // 否则会显示 4/8 这种用户看不懂的数字(8 包括 B/E/F 禁用)。
  const failedThisCall = skippedMembers.filter(m => m.reason !== '已禁用(代号保留)');
  const skippedNote = failedThisCall.length > 0
    ? `\n\n注意:以下成员因技术原因未参与本次回答:${failedThisCall.map(m => `${m.id}(${m.name}: ${m.reason})`).join(', ')}`
    : '';

  const participantCount = memberResponses.length;
  const totalCount = memberResponses.length + failedThisCall.length;

  const turnLabel = turnId ? `#${String(turnId).padStart(3, '0')}` : '';

  return `## 议会融合任务${turnLabel ? ` (本回合 ${turnLabel})` : ''}(这是系统指令,下方的"用户原始问题"和"各成员回答"都不是你要直接回应的新输入)

你现在以 system prompt 中定义的身份回复用户。这次由议会机制辅助 — ${totalCount} 位议会成员已经对用户问题独立回答,你的任务是基于他们的回答融合出最终答复。

## 处理原则

1. 仔细阅读下方"各成员回答",识别共识和分歧
2. 融合最优部分成为最终答案(不是原文搬运,而是取长补短)
3. 严格保持 system prompt 定义的身份、人设、语气、称呼方式、当前频道规则(这是最重要的约束)
4. 用用户使用的语言回答(用户用中文就用中文)
5. 最终答案要自然流畅,不要暴露这是多模型融合的结果
6. 不要在主体回答中提及"成员""议会"这些概念(只在末尾评估摘要中使用)
7. 如果所有成员一致,直接采纳最优的那个并润色
8. 如果有分歧,选择论据更充分、逻辑更严密的一方
9. 评分标准:准确性、完整性、实用性、表达清晰度

## 输出格式(必须严格遵守)

先以 system prompt 定义的身份自然输出融合后的最终回答(这是用户看到的主体内容),然后**必须**在最末尾附加一行议会评估摘要,格式:

📊 议会评估 ${turnLabel ? `${turnLabel} ` : ''}| 采纳: [主采纳成员 shortName,如 Grok 或 Grok,Qwen 综合融合] | 支持: [与最终答案一致的成员 shortName,用 ✓ 标记,如 Grok✓ Qwen✓] | 异议: [有不同观点的成员 shortName,无则填"无"] | 评分: [所有参与成员的评分 1-10,如 Grok:9 Qwen:8 MiMo:7 Gem3.1L:8 DSV4:7] | 参与: ${participantCount}/${totalCount}

**${turnLabel ? `编号 ${turnLabel} 是本议会回合的唯一标识,必须在评估行里出现(放在"议会评估"和"|"之间)`: ''}** — 这个编号让 Arthur 以后可以引用本次议会答案做对比(例如他后续问 "${turnLabel} 那次 Gem3.1L 怎么说",议会代理会基于编号读取 archive 给你看成员原文)。

**评估行硬性要求(违反会被判为失败)**:
- 全部用 shortName(就是上方"### 成员"块标题里那个名字,例如 Grok / Qwen / MiMo / Gem3.1L / DSV4),**不要用单字母代号 A/C/D/H/I**
- "支持"和"异议"两栏加起来必须覆盖**全部 ${participantCount} 个参与成员**,不能有谁既不在支持也不在异议里
- "评分"栏必须列出**全部 ${participantCount} 个参与成员**的分数,不能遗漏任何人
${skippedNote}

---

## 用户原始问题

${originalQuestion}

---

## 各成员回答

${responsesBlock}`;
}

// ─── API Call Helpers ────────────────────────────────────────────────────────

/**
 * Strip thinking tags from model output.
 */
function stripThinkTags(text) {
  if (!text) return text;
  return text.replace(/<think>[\s\S]*?<\/think>\s*/g, '').trim();
}

/**
 * Call an OpenAI-compatible API with timeout.
 * Returns { content, usage } where usage contains token counts.
 */

/**
 * Add cache_control to system messages for OpenRouter prompt caching.
 */
function addCacheControl(messages) {
  return messages.map(m => {
    if (m.role === 'system' && typeof m.content === 'string' && m.content.length > 500) {
      return {
        ...m,
        content: [{ type: 'text', text: m.content, cache_control: { type: 'ephemeral' } }],
      };
    }
    return m;
  });
}

async function callModel(config, messages, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  // DSV4 协议适配层 — 仅在 DeepSeek provider 时应用。
  // 处理 traps #2 (image_url strip) + #7 (reasoning_content strip)。
  // 其他模型(Grok / Qwen / Gem / GPT)不受影响,messages 原样传递。
  const preparedMessages = isDeepSeekProvider(config)
    ? dsv4PrepareMessages(messages)
    : messages;

  try {
    const response = await fetch(`${config.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${config.apiKey()}`,
      },
      body: JSON.stringify({
        model: config.model,
        messages: addCacheControl(preparedMessages),
        max_tokens: 4096,
        temperature: config.fixedTemperature ?? 0.7,
        stream: false,
        ...(config.extraBody || {}),
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      const errText = await response.text().catch(() => 'unknown');
      throw new Error(`HTTP ${response.status}: ${errText.slice(0, 200)}`);
    }

    const data = await response.json();
    let content = data.choices?.[0]?.message?.content;

    // Some models (StepFun) put answer in reasoning field when content is empty
    if (!content && data.choices?.[0]?.message?.reasoning) {
      content = data.choices[0].message.reasoning;
    }

    if (!content) {
      throw new Error('Empty response content');
    }

    // Strip <think> tags from output
    content = stripThinkTags(content);

    // Extract usage stats
    const usage = data.usage || {};

    return { content, usage };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Call a Council member with the user's question.
 */
async function callMember(member, messages) {
  const start = Date.now();
  try {
    const { content, usage } = await callModel(member, messages, MEMBER_TIMEOUT_MS);
    const elapsed = Date.now() - start;
    const tokens = `${usage.prompt_tokens || '?'}in/${usage.completion_tokens || '?'}out${usage.cached_tokens ? '/'+usage.cached_tokens+'cached' : ''}`;
    console.log(`  ✅ ${member.id}(${member.name}): ${elapsed}ms, ${content.length}ch, ${tokens}`);
    return { id: member.id, name: member.name, shortName: member.shortName, content, elapsed, success: true, usage };
  } catch (err) {
    const elapsed = Date.now() - start;
    const reason = err.name === 'AbortError' ? 'timeout' : err.message?.slice(0, 100);
    console.log(`  ❌ ${member.id}(${member.name}): ${elapsed}ms, error: ${reason}`);
    return { id: member.id, name: member.name, shortName: member.shortName, content: null, elapsed, success: false, error: reason, usage: {} };
  }
}

/**
 * Call Chairman to fuse member responses.
 */
async function callChairman(originalQuestion, memberResponses, skippedMembers, originalSystemPrompt, turnId = null) {
  //  修正():system = client-provided system(完整,不 slice),
  // user = 融合任务 + 成员答案。Chairman 天然以 Alfred 身份回应,频道规则和 Memory 不丢。
  // turnId 传入用于 #NNN 编号注入,Chairman 输出评估行带编号
  const systemContent = originalSystemPrompt || '你是一个智能助手。';
  const fusionUserMessage = buildFusionUserMessage(originalQuestion, memberResponses, skippedMembers, turnId);
  const messages = [
    { role: 'system', content: systemContent },
    { role: 'user', content: fusionUserMessage },
  ];

  // Try primary Chairman
  try {
    console.log(`  🏛️ Chairman (${CHAIRMAN_PRIMARY.name})...`);
    const { content, usage } = await callModel(CHAIRMAN_PRIMARY, messages, CHAIRMAN_TIMEOUT_MS);
    console.log(`  ✅ Chairman: ${content.length}ch, ${usage.prompt_tokens || '?'}in/${usage.completion_tokens || '?'}out`);
    return { content, chairman: CHAIRMAN_PRIMARY.name, usage };
  } catch (err) {
    console.log(`  ❌ Chairman primary failed: ${err.message?.slice(0, 100)}`);
  }

  // Fallback to secondary Chairman
  try {
    console.log(`  🏛️ Chairman fallback (${CHAIRMAN_FALLBACK.name})...`);
    const { content, usage } = await callModel(CHAIRMAN_FALLBACK, messages, CHAIRMAN_TIMEOUT_MS);
    console.log(`  ✅ Chairman fallback: ${content.length}ch, ${usage.prompt_tokens || '?'}in/${usage.completion_tokens || '?'}out`);
    return { content, chairman: CHAIRMAN_FALLBACK.name, usage };
  } catch (err) {
    console.log(`  ❌ Chairman fallback also failed: ${err.message?.slice(0, 100)}`);
    return null;
  }
}

// ─── Image Detection ────────────────────────────────────────────────────────

/**
 * Check if the messages contain any image content.
 */
function hasImages(messages) {
  for (const msg of messages) {
    if (Array.isArray(msg.content)) {
      for (const part of msg.content) {
        if (part.type === 'image_url' || part.type === 'image') return true;
      }
    }
  }
  return false;
}

// ─── Logging ────────────────────────────────────────────────────────────────

function logCouncilResult(question, memberResults, chairmanResult, totalElapsed) {
  const timestamp = new Date().toISOString();
  const questionPreview = (typeof question === 'string' ? question : JSON.stringify(question)).slice(0, 100);

  const lines = [
    `\n## ${timestamp}`,
    `**Q:** ${questionPreview}${questionPreview.length >= 100 ? '...' : ''}`,
    `**耗时:** ${totalElapsed}ms | **Chairman:** ${chairmanResult?.chairman || 'FAILED'}`,
  ];

  const succeeded = memberResults.filter(r => r.success);
  const failed = memberResults.filter(r => !r.success);

  lines.push(`**成员:** ${succeeded.map(r => `${r.id}✅(${r.elapsed}ms)`).join(' ')} ${failed.map(r => `${r.id}❌(${r.error})`).join(' ')}`);

  // Token usage summary
  const tokenLines = succeeded.map(r => {
    const u = r.usage || {};
    return `${r.id}:${u.prompt_tokens || '?'}/${u.completion_tokens || '?'}${u.cached_tokens ? '(cached:'+u.cached_tokens+')' : ''}`;
  });
  const chairmanUsage = chairmanResult?.usage || {};
  tokenLines.push(`Chairman:${chairmanUsage.prompt_tokens || '?'}/${chairmanUsage.completion_tokens || '?'}`);
  const totalIn = succeeded.reduce((s, r) => s + (r.usage?.prompt_tokens || 0), 0) + (chairmanUsage.prompt_tokens || 0);
  const totalOut = succeeded.reduce((s, r) => s + (r.usage?.completion_tokens || 0), 0) + (chairmanUsage.completion_tokens || 0);
  lines.push(`**Tokens:** ${tokenLines.join(' | ')} | **合计:** ${totalIn}in/${totalOut}out`);


  // Member response summaries (first 200 chars for review)
  const summaryLines = memberResults.map(r => {
    if (r.success && r.content) {
      const preview = r.content.replace(/\n/g, ' ').slice(0, 200);
      return `  ${r.id}(${r.shortName || r.name}): ${preview}${r.content.length > 200 ? '...' : ''}`;
    } else {
      return `  ${r.id}(${r.shortName || r.name}): [${r.error || 'no content'}]`;
    }
  });
  lines.push(`**\u6458\u8981:**`);
  lines.push(...summaryLines);

  // Extract evaluation summary from Chairman's response
  const evalMatch = chairmanResult?.content?.match(/📊.*$/m);
  if (evalMatch) {
    lines.push(`**评估:** ${evalMatch[0]}`);
  }

  lines.push('');

  try {
    appendFileSync(LOG_PATH, lines.join('\n'));
  } catch (e) {
    console.error('Failed to write council log:', e.message);
  }
}

// ─── Main Council Flow ──────────────────────────────────────────────────────

async function runCouncil(requestBody) {
  const startTime = Date.now();
  const messages = requestBody.messages || [];
  const toolsSchema = requestBody.tools || [];

  // Extract the user's question (last user message)
  const userMessages = messages.filter(m => m.role === 'user');
  const lastUserMsg = userMessages[userMessages.length - 1];
  const originalQuestion = typeof lastUserMsg?.content === 'string'
    ? lastUserMsg.content
    : JSON.stringify(lastUserMsg?.content || '');

  console.log(`\n🏛️ Council session started: "${originalQuestion.slice(0, 80)}..."`);

  // ─── Phase 1: Front-end routing (MiniMax decides: tool or consensus?) ───
  const routeResult = await frontEndRoute(messages, toolsSchema);

  if (routeResult.action === 'tool_call') {
    console.log(`  🏛️ Council routing to tool call (${Date.now() - startTime}ms)`);
    return { type: 'tool_call', response: routeResult.response };
  }

  if (routeResult.action === 'error') {
    return {
      type: 'text',
      content: routeResult.message,
      model: 'council-v1/error',
      totalElapsed: Date.now() - startTime,
    };
  }

  if (routeResult.action === 'limit_reached') {
    return {
      type: 'text',
      content: routeResult.message,
      model: 'council-v1/limit-reached',
      totalElapsed: Date.now() - startTime,
    };
  }

  // ─── Phase 2: Council consensus flow ───

  // Detect images
  const containsImages = hasImages(messages);
  if (containsImages) {
    console.log('  📷 Images detected, Step 3.5 Flash will be skipped');
  }

  // Prepare member messages: replace system prompt
  const systemMsg = messages.find(m => m.role === 'system');
  const memberSystemPrompt = buildMemberSystemPrompt(systemMsg?.content);
  const memberMessages = [
    { role: 'system', content: memberSystemPrompt },
    ...messages.filter(m => m.role !== 'system'),
  ];

  // Filter members based on image presence
  const activeMembers = MEMBERS.filter(m => {
    if (m.enabled === false) return false; // : 跳过禁用成员(B/E/F)
    if (containsImages && !m.supportsVision) return false;
    return true;
  });
  const skippedMembers = MEMBERS
    .filter(m => !activeMembers.includes(m))
    .map(m => {  // Note:reason 按实际原因区分,不再固定写"纯文本模型"
      let reason;
      if (m.enabled === false) reason = '已禁用(代号保留)';
      else if (containsImages && !m.supportsVision) reason = '纯文本模型,图片任务跳过';
      else reason = '其他原因跳过';
      return { id: m.id, name: m.name, shortName: m.shortName, reason };
    });

  // Call all members in parallel
  console.log(`  📡 Dispatching to ${activeMembers.length} members...`);
  const memberResults = await Promise.all(
    activeMembers.map(m => callMember(m, memberMessages))
  );

  const successfulResponses = memberResults.filter(r => r.success);
  const failedResponses = memberResults.filter(r => !r.success);

  // Add failed members to skipped list
  for (const f of failedResponses) {
    skippedMembers.push({ id: f.id, name: f.name, reason: f.error });
  }

  console.log(`  📊 ${successfulResponses.length}/${activeMembers.length} members responded`);

  // Check minimum responses
  if (successfulResponses.length < MIN_RESPONSES) {
    // Not enough responses — if we have at least 1, use it directly
    if (successfulResponses.length === 1) {
      const solo = successfulResponses[0];
      console.log(`  ⚠️ Only 1 member responded, using ${solo.id} directly`);
      const totalElapsed = Date.now() - startTime;
      return {
        type: 'text',
        content: solo.content + `\n\n⚡ 由单一模型 ${solo.name} 回复（议会未达法定人数）`,
        model: `council-v1/${solo.name}`,
        totalElapsed,
      };
    }
    // No responses at all
    throw new Error('Council failed: no members responded');
  }

  // 议会回合编号 — Chairman 输出评估行带 #NNN,Archive 持久化用
  const turnId = getNextCouncilTurnId();
  console.log(`  🏷️  Council turn #${String(turnId).padStart(3, '0')}`);

  // Call Chairman
  const chairmanResult = await callChairman(originalQuestion, successfulResponses, skippedMembers, systemMsg?.content, turnId);
  const totalElapsed = Date.now() - startTime;

  if (!chairmanResult) {
    // Chairman failed — use the longest member response as fallback
    const best = successfulResponses.reduce((a, b) => a.content.length > b.content.length ? a : b);
    console.log(`  ⚠️ Chairman failed, falling back to ${best.id}`);
    logCouncilResult(originalQuestion, memberResults, null, totalElapsed);
    return {
      type: 'text',
      content: best.content + `\n\n⚡ Chairman 不可用，由 ${best.name} 直接回复`,
      model: `council-v1/${best.name}`,
      totalElapsed,
    };
  }

  console.log(`  ✅ Council complete in ${totalElapsed}ms`);
  logCouncilResult(originalQuestion, memberResults, chairmanResult, totalElapsed);

  // 持久化议会成员原文 + Chairman 输出到 archive
  // 让 Arthur 后续能引用 #NNN 编号让 Chairman 拿出来对比
  //  ( 修正):加 skippedMembers 参数,让 archive 存"为什么少人"信息
  writeCouncilArchive(turnId, originalQuestion, successfulResponses, skippedMembers, chairmanResult.content, chairmanResult.chairman);

  return {
    type: 'text',
    content: chairmanResult.content,
    model: `council-v1/chairman-${chairmanResult.chairman}`,
    totalElapsed,
  };
}

// ─── HTTP Server ────────────────────────────────────────────────────────────

const server = createServer(async (req, res) => {
  // Health check
  if (req.method === 'GET' && (req.url === '/' || req.url === '/health')) {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'ok', service: 'council-proxy', version: '3.0.0' }));
    return;
  }

  // Models list (for OpenAI compat discovery)
  // 议会自我介绍接口系统化 — 不写死值,从成员/Chairman 配置动态聚合
  // 算法:输入容量 = 所有 enabled 成员 + Chairman 里最小那个(议会受瓶颈限制);
  //      识图 = 至少一个 enabled 成员能识图(代理对不识图成员 strip image);
  //      最大输出 = Chairman 的 maxOutputTokens(议会最终回答由 Chairman 生成)
  // 优势:换 Chairman / 加减成员 / 改议会架构,这段无需再改
  if (req.method === 'GET' && req.url === '/v1/models') {
    const activeMembers = MEMBERS.filter(m => m.enabled !== false);
    const allActive = [...activeMembers, CHAIRMAN_PRIMARY];
    const contextLength = Math.min(...allActive.map(m => m.contextLength));
    const supportsVision = activeMembers.some(m => m.supportsVision);
    const maxCompletionTokens = CHAIRMAN_PRIMARY.maxOutputTokens;
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      object: 'list',
      data: [{
        id: 'council-v1',
        object: 'model',
        created: 1745568000,
        owned_by: 'council-proxy',
        context_length: contextLength,
        max_completion_tokens: maxCompletionTokens,
        supports_function_calling: true,
        supports_vision: supportsVision,
      }],
    }));
    return;
  }

  // Main endpoint
  if (req.method === 'POST' && req.url === '/v1/chat/completions') {
    let body = '';
    for await (const chunk of req) body += chunk;

    try {
      const requestBody = JSON.parse(body);
      const wantStream = requestBody.stream === true;

      // Run the council (may return text or tool_call)
      const result = await runCouncil(requestBody);

      // Tool call → return tool_call response to Gateway
      if (result.type === 'tool_call') {
        if (wantStream) {
          // SSE streaming format for tool_calls
          res.writeHead(200, {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
          });
          const tc = result.response;
          const sseId = tc.id || `council-${Date.now()}`;
          const created = tc.created || Math.floor(Date.now() / 1000);
          const model = tc.model || 'council-v1/router';
          const toolCalls = tc.choices?.[0]?.message?.tool_calls || [];

          // Add index to each tool_call for delta format
          const deltaToolCalls = toolCalls.map((t, i) => ({
            index: i, id: t.id, type: t.type, function: t.function,
          }));

          // Chunk 1: role + tool_calls
          res.write(`data: ${JSON.stringify({
            id: sseId, object: 'chat.completion.chunk', created, model,
            choices: [{ index: 0, delta: { role: 'assistant', tool_calls: deltaToolCalls }, finish_reason: null }],
          })}\n\n`);

          // Chunk 2: finish
          res.write(`data: ${JSON.stringify({
            id: sseId, object: 'chat.completion.chunk', created, model,
            choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }],
          })}\n\n`);

          // End
          res.write('data: [DONE]\n\n');
          res.end();
          console.log(`  ✅ Sent SSE streaming tool_call (${toolCalls.length} calls)`);
        } else {
          // Non-streaming: original JSON format
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(result.response));
        }
        return;
      }

      // Text response
      if (wantStream) {
        // SSE streaming format for the host application compatibility
        res.writeHead(200, {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
        });
        const sseId = `council-${Date.now()}`;
        const created = Math.floor(Date.now() / 1000);

        // Chunk 1: role
        res.write(`data: ${JSON.stringify({
          id: sseId, object: 'chat.completion.chunk', created,
          model: result.model,
          choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }],
        })}\n\n`);

        // Chunk 2: content
        res.write(`data: ${JSON.stringify({
          id: sseId, object: 'chat.completion.chunk', created,
          model: result.model,
          choices: [{ index: 0, delta: { content: result.content }, finish_reason: null }],
        })}\n\n`);

        // Chunk 3: finish
        res.write(`data: ${JSON.stringify({
          id: sseId, object: 'chat.completion.chunk', created,
          model: result.model,
          choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
          usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
        })}\n\n`);

        // End
        res.write('data: [DONE]\n\n');
        res.end();
        console.log(`  ✅ Sent SSE streaming response (${result.content.length} chars)`);
      } else {
        // Non-streaming: original JSON format
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          id: `council-${Date.now()}`,
          object: 'chat.completion',
          created: Math.floor(Date.now() / 1000),
          model: result.model,
          choices: [{
            index: 0,
            message: { role: 'assistant', content: result.content },
            finish_reason: 'stop',
          }],
          usage: {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
          },
        }));
      }
    } catch (err) {
      console.error('Council error:', err);
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        error: { message: err.message, type: 'council_error' },
      }));
    }
    return;
  }

  // 404
  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: { message: 'Not found' } }));
});

server.listen(PORT, HOST, () => {
  console.log(`🏛️ Council Proxy v3.0.0 (OpenRouter + Router) listening on ${HOST}:${PORT}`);
  console.log(`   Members: ${MEMBERS.map(m => `${m.id}(${m.name})${m.enabled === false ? '[DISABLED]' : ''}`).join(', ')}`);
  console.log(`   Active:  ${MEMBERS.filter(m => m.enabled !== false).map(m => m.id).join('/')}`);
  console.log(`   Chairman: ${CHAIRMAN_PRIMARY.name} (fallback: ${CHAIRMAN_FALLBACK.name})`);
  console.log(`   Timeouts: member=${MEMBER_TIMEOUT_MS}ms, chairman=${CHAIRMAN_TIMEOUT_MS}ms`);
  console.log(`   Router: OpenRouter (single API key)`);
  console.log(`   Log: ${LOG_PATH}`);

  if (!OPENROUTER_KEY) {
    console.warn(`   ⚠️ OPENROUTER_API_KEY (OpenRouter) is not set!`);
  }
  if (!DEEPSEEK_KEY) {
    console.warn(`   ⚠️ DEEPSEEK_API_KEY (DeepSeek direct) is not set — I(DSV4) will fail.`);
  }
});

server.on('error', (err) => {
  console.error('Server error:', err);
  process.exit(1);
});
