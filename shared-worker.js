import {
    AutoProcessor,
    Gemma4ForConditionalGeneration,
    TextStreamer
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';
  
const connections = [];

// Estado global do worker para sincronizar todas as abas
let state = { status: 'idle', label: 'Nao carregado', progress: 0 };
let processor = null;
let model = null;
let isRunning = false;

// Função para avisar TODAS as abas abertas sobre qualquer mudança
function broadcast(msg) {
    connections.forEach(port => {
        try { port.postMessage(msg); } catch (e) {}
    });
}

function updateState(status, label, progress = 0) {
    state = { status, label, progress };
    broadcast({ type: 'state', state });
}

self.onconnect = function(e) {
    const port = e.ports[0];
    connections.push(port);

    // Assim que uma nova aba conecta, avisamos em qual estado o modelo está
    port.postMessage({ type: 'state', state });

    port.onmessage = async function(event) {
        const { action, payload } = event.data;

        if (action === 'load_model') {
            // Ignora se já estiver carregando ou inferindo
            if (state.status === 'loading' || state.status === 'running') return;
            loadModel(payload.modelId);
        } 
        else if (action === 'extract') {
            // Só roda se o modelo estiver pronto
            if (state.status !== 'ready' || isRunning) return;
            runExtract(payload);
        }
    };
};

async function loadModel(modelId) {
    isRunning = false;
    updateState('loading', 'Carregando...');

    try {
        let lastPct = 0;
        const onProgress = (info) => {
            if (info.status === 'progress' || info.status === 'progress_total') {
                const pct = Math.round(info.progress || 0);
                if (pct > lastPct) lastPct = pct;
                const label = info.status === 'progress_total' ? `Carregando... ${lastPct}%` : `Baixando... ${lastPct}%`;
                updateState('loading', label, lastPct);
            } else if (info.status === 'initiate') {
                updateState('loading', `Iniciando: ${info.file || ''}`, lastPct);
            } else if (info.status === 'loading') {
                updateState('loading', 'Carregando pesos...', 100);
            } else if (info.status === 'done') {
                updateState('loading', 'Processando...', 100);
            }
        };

        processor = await AutoProcessor.from_pretrained(modelId);

        model = await Gemma4ForConditionalGeneration.from_pretrained(modelId, {
            dtype: 'q4f16',
            device: 'webgpu',
            progress_callback: onProgress,
        });

        updateState('ready', 'Pronto OK');

    } catch(e) {
        console.error(e);
        updateState('error', 'Erro');
        broadcast({ type: 'error', message: e.message || 'Falha ao carregar o modelo. Verifique se seu browser suporta WebGPU.' });
    }
}

async function runExtract({ text, systemPrompt, temperature, maxTokens }) {
    isRunning = true;
    updateState('running', 'Inferindo...');

    try {
        const messages = [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: `Texto:\n${text}` }
        ];

        const promptStr = processor.apply_chat_template(messages, {
            enable_thinking: false,
            add_generation_prompt: true,
        });

        const inputs = await processor(promptStr, null, null, {
            add_special_tokens: false,
        });

        let generated = '';

        const streamer = new TextStreamer(processor.tokenizer, {
            skip_prompt: true,
            skip_special_tokens: false,
            callback_function: (chunk) => {
                generated += chunk;
                // Envia o output do stream em tempo real para quem pediu
                broadcast({ type: 'stream', chunk: generated });
            }
        });

        // Configura a inferência com os parâmetros recebidos do UI
        const outputs = await model.generate({
            ...inputs,
            max_new_tokens: parseInt(maxTokens, 10),
            do_sample: parseFloat(temperature) > 0,
            temperature: parseFloat(temperature) > 0 ? parseFloat(temperature) : undefined,
            streamer,
        });

        const inputLen = inputs.input_ids.dims.at(-1);
        const decoded = processor.batch_decode(
            outputs.slice(null, [inputLen, null]),
            { skip_special_tokens: true }
        );
        generated = decoded[0] || generated;

        // Envia o resultado final
        broadcast({ type: 'result', result: generated });

    } catch(e) {
        console.error(e);
        broadcast({ type: 'error', message: e.message || 'Erro durante a inferencia.' });
    } finally {
        isRunning = false;
        updateState('ready', 'Pronto OK');
    }
}
