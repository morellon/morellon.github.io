// ner-worker.js  -  SharedWorker
// Owns one model instance shared across all tabs.
// Protocol (postMessage):
//   TAB -> WORKER  { id, type, payload }
//   WORKER -> TAB  { id, type, payload }   (id echoed for request/response matching)
//
// Types TAB->WORKER:  load | extract | ping
// Types WORKER->TAB:  progress | ready | error | token | result | state | pong

import {
  AutoProcessor,
  Gemma4ForConditionalGeneration,
  TextStreamer
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';

// -- STATE -------------------------------------------------------------
const ports   = new Set();   // all connected tabs
let processor = null;
let model     = null;
let modelId   = '';
let loadState = 'idle';      // idle | loading | ready | error
let loadError = '';

// Inference queue: only one job at a time (GPU is single-tenant)
const queue   = [];
let   busy    = false;

// -- HELPERS -----------------------------------------------------------
function broadcast(msg) {
  ports.forEach(p => p.postMessage(msg));
}

function send(port, msg) {
  port.postMessage(msg);
}

// -- CONNECTION --------------------------------------------------------
self.onconnect = (e) => {
  const port = e.ports[0];
  ports.add(port);

  // Immediately send current state to the new tab
  send(port, {
    type: 'state',
    payload: { loadState, modelId, loadError }
  });

  port.onmessage = async (ev) => {
    const { id, type, payload } = ev.data;

    if (type === 'ping') {
      send(port, { id, type: 'pong', payload: { loadState, modelId } });

    } else if (type === 'load') {
      await handleLoad(payload.modelId);

    } else if (type === 'extract') {
      queue.push({ id, port, payload });
      drainQueue();
    }
  };

  port.onmessageerror = () => {};

  // Remove port on disconnect (best-effort)
  port.addEventListener('close', () => ports.delete(port));
  port.start();
};

// -- LOAD --------------------------------------------------------------
async function handleLoad(mid) {
  if (loadState === 'loading') return;   // already in progress
  if (loadState === 'ready' && modelId === mid) {
    // Already loaded, just tell everyone
    broadcast({ type: 'ready', payload: { modelId: mid } });
    return;
  }

  loadState = 'loading';
  modelId   = mid;
  loadError = '';
  processor = null;
  model     = null;

  broadcast({ type: 'state', payload: { loadState, modelId, loadError } });

  // Per-file progress tracking (downloads are parallel)
  const fileProgress = new Map();
  let gpuPhase = false;

  function calcOverall() {
    let sumL = 0, sumT = 0;
    fileProgress.forEach(f => { sumL += f.loaded; sumT += f.total; });
    return sumT > 0 ? Math.min(Math.round(sumL / sumT * 100), 99) : 0;
  }

  let rafHandle = null;
  function scheduleProgress() {
    if (rafHandle || gpuPhase) return;
    // SharedWorker has no requestAnimationFrame -- use setTimeout 60fps equivalent
    rafHandle = setTimeout(() => {
      rafHandle = null;
      if (gpuPhase) return;
      const overall = calcOverall();
      let activeFile = '', bestPct = -1;
      fileProgress.forEach((f, name) => {
        if (f.total > 0 && f.loaded < f.total) {
          const pct = Math.round(f.loaded / f.total * 100);
          if (pct > bestPct) { bestPct = pct; activeFile = name; }
        }
      });
      broadcast({
        type: 'progress',
        payload: { overall, activeFile, activePct: bestPct < 0 ? 0 : bestPct, phase: 'download' }
      });
    }, 16);
  }

  const onProgress = (info) => {
    const fname = (info.file || '').split('/').pop();
    if (info.status === 'initiate') {
      fileProgress.set(fname, { loaded: 0, total: info.total || 1 });
    } else if (info.status === 'progress') {
      const total  = info.total  || 1;
      const loaded = info.loaded || Math.round((info.progress || 0) / 100 * total);
      const prev   = fileProgress.get(fname) || { loaded: 0, total };
      fileProgress.set(fname, { loaded: Math.max(loaded, prev.loaded), total: Math.max(total, 1) });
      scheduleProgress();
    } else if (info.status === 'done') {
      const prev = fileProgress.get(fname) || { loaded: 1, total: 1 };
      fileProgress.set(fname, { loaded: prev.total, total: prev.total });
      scheduleProgress();
    } else if (info.status === 'progress_total') {
      gpuPhase = true;
      broadcast({ type: 'progress', payload: { overall: Math.round(info.progress || 0), phase: 'gpu' } });
    } else if (info.status === 'loading') {
      gpuPhase = true;
      broadcast({ type: 'progress', payload: { overall: 99, phase: 'gpu', activeFile: 'GPU init' } });
    }
  };

  try {
    processor = await AutoProcessor.from_pretrained(mid, { progress_callback: onProgress });
    model     = await Gemma4ForConditionalGeneration.from_pretrained(mid, {
      dtype: 'q4f16',
      device: 'webgpu',
      progress_callback: onProgress,
    });

    loadState = 'ready';
    broadcast({ type: 'ready', payload: { modelId: mid } });
    broadcast({ type: 'state', payload: { loadState, modelId, loadError } });

    // Drain any queued requests that came in while loading
    drainQueue();

  } catch (err) {
    loadState = 'error';
    loadError = err.message || String(err);
    broadcast({ type: 'error', payload: { message: loadError } });
    broadcast({ type: 'state', payload: { loadState, modelId, loadError } });
  }
}

// -- INFERENCE QUEUE ---------------------------------------------------
function drainQueue() {
  if (busy || queue.length === 0 || loadState !== 'ready') return;
  busy = true;
  const job = queue.shift();
  runInference(job).finally(() => {
    busy = false;
    drainQueue();
  });
}

async function runInference({ id, port, payload }) {
  const { text, config } = payload;

  const PROMPT_SYSTEM = config.systemPrompt || DEFAULT_SYSTEM_PROMPT;

  const messages = [
    { role: 'system', content: PROMPT_SYSTEM },
    { role: 'user',   content: 'Texto:\n' + text }
  ];

  try {
    const prompt = processor.apply_chat_template(messages, {
      enable_thinking: false,
      add_generation_prompt: true,
    });

    const inputs = await processor(prompt, null, null, { add_special_tokens: false });

    let generated = '';

    const streamer = new TextStreamer(processor.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: false,
      callback_function: (chunk) => {
        generated += chunk;
        // Send token to requesting tab only
        send(port, { id, type: 'token', payload: { chunk, generated } });
        // Also broadcast queue depth to all tabs
        broadcast({ type: 'queue', payload: { depth: queue.length, busy: true } });
      }
    });

    const outputs = await model.generate({
      ...inputs,
      max_new_tokens:    config.maxNewTokens    || 512,
      temperature:       config.temperature     || 1.0,
      top_p:             config.topP            || 1.0,
      repetition_penalty:config.repetitionPenalty || 1.0,
      do_sample:         config.doSample        || false,
      streamer,
    });

    // Decode final output
    const inputLen = inputs.input_ids.dims.at(-1);
    const decoded  = processor.batch_decode(
      outputs.slice(null, [inputLen, null]),
      { skip_special_tokens: true }
    );
    generated = decoded[0] || generated;

    send(port, { id, type: 'result', payload: { generated } });

  } catch (err) {
    send(port, { id, type: 'error', payload: { message: err.message || String(err) } });
  }

  broadcast({ type: 'queue', payload: { depth: queue.length, busy: false } });
}

// -- DEFAULT PROMPT ----------------------------------------------------
const DEFAULT_SYSTEM_PROMPT = [
  'Voce e um extrator de entidades nomeadas especializado em textos brasileiros.',
  'Retorne SOMENTE um objeto JSON valido, sem markdown, sem explicacoes, sem texto fora do JSON.',
  'Formato EXATO (todos os campos obrigatorios):',
  '{"pessoa":[],"empresa":[],"endereco":[],"cidade":[],"estado":[],"pais":[],"nascimento":[],"cpf":[],"cnpj":[],"rastreio":[],"cte":[],"nfe":[]}',
  'Regras: preserve valores exatamente como aparecem no texto original; sem duplicatas por chave; arrays vazios se nao encontrado.'
].join('\n');
