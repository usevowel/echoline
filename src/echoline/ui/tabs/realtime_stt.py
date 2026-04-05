import logging

import gradio as gr

from echoline.config import Config

logger = logging.getLogger(__name__)

JS_CODE = """
(function() {
    var STORAGE_KEY = 'echoline_api_key';

    function getApiKey() {
        try {
            return localStorage.getItem(STORAGE_KEY) || '';
        } catch (e) {
            return '';
        }
    }

    function getWsUrl() {
        var proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return proto + '//' + window.location.host + '/v1/realtime';
    }

    var ws = null;
    var audioContext = null;
    var mediaStream = null;
    var scriptNode = null;
    var isRecording = false;
    var transcriptStartTime = null;
    var transcriptCount = 0;
    var accumulatedTranscript = '';

    var connectBtn = null;
    var stopBtn = null;
    var clearBtn = null;
    var modelSelect = null;
    var vadIndicator = null;
    var connectionStatus = null;
    var transcriptArea = null;

    function initComponents() {
        var selects = document.querySelectorAll('select');
        for (var i = 0; i < selects.length; i++) {
            if (selects[i].querySelector('optgroup')) {
                modelSelect = selects[i];
                break;
            }
        }

        var labels = document.querySelectorAll('.gradio-container label');
        for (var i = 0; i < labels.length; i++) {
            if (labels[i].textContent === 'VAD Status') {
                vadIndicator = labels[i].parentElement;
                break;
            }
        }

        connectionStatus = document.querySelector('[id*="connection-status"], [id*="Connection"]');

        var textareas = document.querySelectorAll('textarea');
        for (var i = 0; i < textareas.length; i++) {
            if (textareas[i].placeholder && textareas[i].placeholder.includes('Connect')) {
                transcriptArea = textareas[i];
                break;
            }
        }
    }

    function setVadState(state) {
        if (!vadIndicator) return;
        var label = vadIndicator.querySelector('.value');
        if (label) {
            label.textContent = state.charAt(0).toUpperCase() + state.slice(1);
        }
    }

    function setConnectionStatus(status, isError) {
        if (connectionStatus) {
            connectionStatus.textContent = status;
            if (isError) {
                connectionStatus.style.color = '#dc2626';
            } else if (status === 'Connected') {
                connectionStatus.style.color = '#16a34a';
            }
        }
    }

    function updateTranscript(text) {
        if (!transcriptArea) return;
        transcriptArea.value = text;
        transcriptArea.dispatchEvent(new Event('input', { bubbles: true }));
    }

    function addTranscript(text) {
        accumulatedTranscript += text + '\\n';
        updateTranscript(accumulatedTranscript);
    }

    function clearTranscripts() {
        accumulatedTranscript = '';
        updateTranscript('');
    }

    function floatTo16BitPCM(float32Array) {
        var buffer = new ArrayBuffer(float32Array.length * 2);
        var view = new DataView(buffer);
        for (var i = 0; i < float32Array.length; i++) {
            var s = Math.max(-1, Math.min(1, float32Array[i]));
            view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
        return buffer;
    }

    function arrayBufferToBase64(buffer) {
        var binary = '';
        var bytes = new Uint8Array(buffer);
        for (var i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }

    async function startRecording() {
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 24000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });

            audioContext = new AudioContext({ sampleRate: 24000 });
            var source = audioContext.createMediaStreamSource(mediaStream);

            scriptNode = audioContext.createScriptProcessor(4096, 1, 1);
            scriptNode.onaudioprocess = function(e) {
                if (!isRecording || !ws || ws.readyState !== WebSocket.OPEN) return;

                var inputData = e.inputBuffer.getChannelData(0);
                var pcm16 = floatTo16BitPCM(inputData);
                var base64Audio = arrayBufferToBase64(pcm16);

                ws.send(JSON.stringify({
                    type: 'input_audio_buffer.append',
                    audio: base64Audio
                }));
            };

            source.connect(scriptNode);
            scriptNode.connect(audioContext.destination);
            isRecording = true;
            transcriptStartTime = Date.now();

        } catch (err) {
            console.error('Microphone access error:', err);
            setConnectionStatus('Mic access denied', true);
            setVadState('error');
            cleanup();
        }
    }

    function stopRecording() {
        isRecording = false;
        if (scriptNode) {
            scriptNode.disconnect();
            scriptNode = null;
        }
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }
        if (mediaStream) {
            mediaStream.getTracks().forEach(function(track) { track.stop(); });
            mediaStream = null;
        }
    }

    function connect() {
        if (!modelSelect) {
            initComponents();
            setTimeout(connect, 100);
            return;
        }

        var model = modelSelect.value;
        if (!model) {
            setConnectionStatus('Select a model first', true);
            return;
        }

        setConnectionStatus('Connecting...', false);

        var wsUrl = getWsUrl();
        var apiKey = getApiKey();

        var url = wsUrl + '?model=' + encodeURIComponent(model) + '&intent=transcription';
        if (apiKey) {
            url += '&api_key=' + encodeURIComponent(apiKey);
        }

        ws = new WebSocket(url);

        ws.onopen = function() {
            console.log('WebSocket connected');
            setConnectionStatus('Connected', false);

            ws.send(JSON.stringify({
                type: 'session.update',
                session: {
                    input_audio_transcription: {
                        model: model
                    },
                    turn_detection: {
                        type: 'server_vad',
                        threshold: 0.9,
                        prefix_padding_ms: 0,
                        silence_duration_ms: 1500,
                        create_response: false
                    }
                }
            }));

            startRecording();
        };

        ws.onmessage = function(event) {
            try {
                var data = JSON.parse(event.data);
                handleMessage(data);
            } catch (e) {
                console.error('Failed to parse message:', e);
            }
        };

        ws.onerror = function(error) {
            console.error('WebSocket error:', error);
            setConnectionStatus('Connection error', true);
            setVadState('error');
        };

        ws.onclose = function(event) {
            console.log('WebSocket closed:', event.code, event.reason);
            setConnectionStatus('Disconnected', event.code !== 1000);
            cleanup();
        };
    }

    function handleMessage(data) {
        switch (data.type) {
            case 'session.created':
            case 'session.updated':
                console.log('Session ready:', data.type);
                break;

            case 'input_audio_buffer.speech_started':
                setVadState('speaking');
                break;

            case 'input_audio_buffer.speech_stopped':
                setVadState('idle');
                break;

            case 'conversation.item.input_audio_transcription.completed':
                if (data.transcript && data.transcript.trim()) {
                    addTranscript(data.transcript.trim());
                }
                break;

            case 'error':
                console.error('Server error:', data);
                setConnectionStatus('Error: ' + (data.error ? data.error.message : 'Unknown'), true);
                setVadState('error');
                break;

            default:
                break;
        }
    }

    function cleanup() {
        stopRecording();
        if (ws) {
            ws = null;
        }
        if (modelSelect && modelSelect.querySelector('option')) {
            modelSelect.disabled = false;
        }
    }

    function disconnect() {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.close(1000, 'User disconnect');
        } else {
            cleanup();
        }
    }

    async function loadModels() {
        try {
            var apiKey = getApiKey();
            var baseUrl = window.location.origin;
            var headers = {};
            if (apiKey) {
                headers['Authorization'] = 'Bearer ' + apiKey;
            }

            var response = await fetch(baseUrl + '/v1/models', { headers: headers });
            if (!response.ok) throw new Error('Failed to fetch models');

            var data = await response.json();
            var models = data.data || [];

            if (!modelSelect) {
                initComponents();
            }

            if (!modelSelect) return;

            modelSelect.innerHTML = '';

            var sttModels = models.filter(function(m) {
                return m.id.includes('whisper') || m.id.includes('faster') || m.id.includes('Systran');
            });
            var otherModels = models.filter(function(m) {
                return !m.id.includes('whisper') && !m.id.includes('faster') && !m.id.includes('Systran');
            });

            if (sttModels.length > 0) {
                var optgroup = document.createElement('optgroup');
                optgroup.label = 'STT Models (Recommended)';
                sttModels.forEach(function(m) {
                    var opt = document.createElement('option');
                    opt.value = m.id;
                    opt.textContent = m.id;
                    optgroup.appendChild(opt);
                });
                modelSelect.appendChild(optgroup);
            }

            if (otherModels.length > 0) {
                var optgroup = document.createElement('optgroup');
                optgroup.label = 'Other Models';
                otherModels.forEach(function(m) {
                    var opt = document.createElement('option');
                    opt.value = m.id;
                    opt.textContent = m.id;
                    optgroup.appendChild(opt);
                });
                modelSelect.appendChild(optgroup);
            }

            if (models.length > 0) {
                modelSelect.value = models[0].id;
                modelSelect.disabled = false;
            }

        } catch (err) {
            console.error('Failed to load models:', err);
        }
    }

    function setupButtons() {
        var buttons = document.querySelectorAll('button');
        for (var i = 0; i < buttons.length; i++) {
            var btn = buttons[i];
            if (btn.textContent === 'Connect' || btn.innerText === 'Connect') {
                connectBtn = btn;
            } else if (btn.textContent === 'Stop' || btn.innerText === 'Stop') {
                stopBtn = btn;
            } else if (btn.textContent === 'Clear Transcript' || btn.innerText === 'Clear Transcript') {
                clearBtn = btn;
            }
        }

        if (connectBtn && stopBtn && clearBtn && modelSelect) {
            connectBtn.addEventListener('click', function() {
                connectBtn.disabled = true;
                stopBtn.disabled = false;
                modelSelect.disabled = true;
                connect();
            });
            stopBtn.addEventListener('click', function() {
                disconnect();
                connectBtn.disabled = false;
                stopBtn.disabled = true;
                modelSelect.disabled = false;
            });
            clearBtn.addEventListener('click', clearTranscripts);
        } else {
            setTimeout(setupButtons, 100);
        }
    }

    setTimeout(function() {
        initComponents();
        setupButtons();
        loadModels();
    }, 500);
})();
"""


def create_realtime_stt_tab(config: Config, api_key_input: gr.Textbox) -> None:
    with gr.Tab(label="Realtime STT"):
        gr.Markdown("### Realtime Speech-to-Text Tester")
        gr.Markdown(
            "Test the realtime transcription API directly in your browser. "
            "Click **Connect** to start capturing microphone audio. "
            "The server will detect speech (VAD) and return transcriptions in real-time."
        )

        with gr.Row():
            model_select = gr.Dropdown(choices=[], label="Model", value="")
            connect_btn = gr.Button("Connect", variant="primary")
            stop_btn = gr.Button("Stop", variant="stop", interactive=False)

        with gr.Row():
            vad_indicator = gr.Label(label="VAD Status", value="Idle")
            connection_status = gr.Label(label="Connection Status", value="Disconnected")

        transcript_area = gr.Textbox(
            label="Transcript",
            lines=15,
            interactive=False,
            placeholder="Click 'Connect' to start realtime transcription...",
        )

        with gr.Row():
            clear_btn = gr.Button("Clear Transcript")

        gr.HTML(f"<script>{JS_CODE}</script>")