"use strict";

const $ = (id) => document.getElementById(id);
const socket = io({ reconnectionDelayMax: 4000 });
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const state = {
    room: null,
    mySid: null,
    peerSid: null,
    signer: null,
    signMode: false,
    modelReady: false,
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    stream: null,
    pc: null,
    pendingCandidates: [],
    wantsRoom: null,
};

// ---------- small helpers ----------

function showAlert(key, message, kind) {
    let el = document.querySelector(`[data-key="${key}"]`);
    if (!el) {
        el = document.createElement("p");
        el.className = "alert" + (kind === "info" ? " info" : "");
        el.dataset.key = key;
        $("alerts").appendChild(el);
    }
    el.textContent = message;
}

function clearAlert(key) {
    const el = document.querySelector(`[data-key="${key}"]`);
    if (el) el.remove();
}

function addLog(who, text) {
    const li = document.createElement("li");
    const b = document.createElement("b");
    b.textContent = who;
    const span = document.createElement("span");
    span.textContent = text;
    li.append(b, span);
    $("logList").prepend(li);
}

function setText(el, text, emptyText) {
    el.textContent = "";
    if (text) {
        el.textContent = text;
    } else {
        const s = document.createElement("span");
        s.className = "empty";
        s.textContent = emptyText;
        el.appendChild(s);
    }
}

function iAmSigner() {
    return state.signMode && state.signer === state.mySid;
}

// ---------- connection status ----------

function setConnection(text, ok) {
    const el = $("connection");
    el.textContent = text;
    el.dataset.ok = String(ok);
}

socket.on("connect", () => {
    setConnection("Connected", true);
    if (state.wantsRoom) {
        // The server forgot us when the connection dropped, so join again.
        resetPeer();
        socket.emit("join", { room: state.wantsRoom });
    }
});

socket.on("disconnect", () => setConnection("Connection lost. Reconnecting", false));
socket.on("connect_error", () => setConnection("Cannot reach the server. Retrying", false));
socket.on("error_message", (data) => showAlert("server", data.message));

fetch("/config").then((r) => r.json()).then((cfg) => {
    state.iceServers = cfg.iceServers;
    state.modelReady = cfg.modelReady;
}).catch(() => showAlert("config", "Could not load settings from the server.", "info"));

fetch("/available_phrases").then((r) => r.json()).then((list) => {
    $("phraseList").textContent = list.length ? list.map((p) => p.toLowerCase()).join(", ") : "none recorded";
}).catch(() => { $("phraseList").textContent = "unknown"; });

// ---------- joining ----------

async function getMedia() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showAlert("media", "This page cannot use the camera here. Camera access only works at http://localhost or over HTTPS. You can still join and watch and read.");
        return null;
    }
    const attempts = [
        [{ video: true, audio: true }, null],
        [{ video: false, audio: true }, "Camera not available. You joined with the microphone only, so sign mode is off for you."],
        [{ video: true, audio: false }, "Microphone not available. You joined with the camera only, so speech recognition is off for you."],
    ];
    let lastError = null;
    for (const [constraints, warning] of attempts) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            if (warning) showAlert("media", warning, "info");
            return stream;
        } catch (err) {
            lastError = err;
        }
    }
    const blocked = lastError && lastError.name === "NotAllowedError";
    showAlert("media", blocked
        ? "The camera and microphone are blocked. Allow them from the lock icon in the address bar, then leave and join again. You joined without them."
        : "No camera or microphone was found. You joined without them.");
    return null;
}

$("joinForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const room = $("roomInput").value.trim();
    if (!/^[A-Za-z0-9_-]{1,40}$/.test(room)) {
        showAlert("join", "Room names use letters, digits, - and _ (up to 40 characters).");
        return;
    }
    clearAlert("join");
    $("joinBtn").disabled = true;
    state.stream = await getMedia();
    if (state.stream) {
        $("localVideo").srcObject = state.stream;
    } else {
        $("localNote").hidden = false;
        $("localNote").textContent = "No camera or microphone.";
    }
    state.wantsRoom = room;
    socket.emit("join", { room });
});

socket.on("join_error", (data) => {
    state.wantsRoom = null;
    $("joinBtn").disabled = false;
    stopStream();
    showAlert("join", data.message);
});

socket.on("joined", (data) => {
    clearAlert("join");
    state.room = data.room;
    state.mySid = data.sid;
    state.peerSid = data.peers[0] || null;
    state.modelReady = data.modelReady;
    applySignMode(data.signMode, data.signer);
    newPeerConnection();
    $("join").hidden = true;
    $("call").hidden = false;
    $("signBtn").focus();
    renderRoom();
});

$("leaveBtn").addEventListener("click", () => {
    state.wantsRoom = null;
    socket.emit("leave");
    resetPeer();
    stopHandTracking();
    stopSpeech();
    stopStream();
    state.room = state.peerSid = state.signer = null;
    state.signMode = false;
    $("call").hidden = true;
    $("join").hidden = false;
    $("joinBtn").disabled = false;
    $("roomInput").focus();
});

function stopStream() {
    if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
    $("localVideo").srcObject = null;
}

// ---------- WebRTC ----------

function newPeerConnection() {
    resetPeer();
    const pc = new RTCPeerConnection({ iceServers: state.iceServers });
    state.pc = pc;
    const kinds = new Set();
    if (state.stream) {
        state.stream.getTracks().forEach((t) => { pc.addTrack(t, state.stream); kinds.add(t.kind); });
    }
    ["audio", "video"].forEach((kind) => {
        if (!kinds.has(kind)) pc.addTransceiver(kind, { direction: "recvonly" });
    });
    pc.ontrack = (e) => {
        $("remoteVideo").srcObject = e.streams[0];
        $("remoteNote").hidden = true;
    };
    pc.onicecandidate = (e) => {
        if (e.candidate) socket.emit("signal", { type: "candidate", candidate: e.candidate });
    };
    pc.onconnectionstatechange = () => {
        if (pc !== state.pc) return;
        if (pc.connectionState === "failed") {
            showAlert("rtc", "The video connection failed. On different networks this needs a TURN server, see the README. On the same network, check that a firewall is not blocking it.");
        } else if (pc.connectionState === "connected") {
            clearAlert("rtc");
        }
    };
    return pc;
}

function resetPeer() {
    if (state.pc) {
        state.pc.onconnectionstatechange = null;
        state.pc.close();
    }
    state.pc = null;
    state.pendingCandidates = [];
    $("remoteVideo").srcObject = null;
    $("remoteNote").hidden = false;
}

socket.on("peer_joined", async (data) => {
    state.peerSid = data.sid;
    renderRoom();
    // The person who was already here makes the offer.
    const pc = newPeerConnection();
    try {
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        socket.emit("signal", { type: "offer", sdp: pc.localDescription });
    } catch (err) {
        showAlert("rtc", "Could not start the video connection: " + err.message);
    }
});

socket.on("peer_left", () => {
    state.peerSid = null;
    newPeerConnection();
    renderRoom();
    addLog("Room", "The other person left.");
});

async function flushCandidates() {
    const pending = state.pendingCandidates;
    state.pendingCandidates = [];
    for (const c of pending) {
        try { await state.pc.addIceCandidate(c); } catch (err) { /* stale candidate */ }
    }
}

socket.on("signal", async (data) => {
    try {
        if (!state.pc) newPeerConnection();
        const pc = state.pc;
        if (data.type === "offer") {
            await pc.setRemoteDescription(data.sdp);
            const answer = await pc.createAnswer();
            await pc.setLocalDescription(answer);
            socket.emit("signal", { type: "answer", sdp: pc.localDescription });
            await flushCandidates();
        } else if (data.type === "answer") {
            await pc.setRemoteDescription(data.sdp);
            await flushCandidates();
        } else if (data.type === "candidate" && data.candidate) {
            if (pc.remoteDescription) await pc.addIceCandidate(data.candidate);
            else state.pendingCandidates.push(data.candidate);
        }
    } catch (err) {
        showAlert("rtc", "Video connection error: " + err.message);
    }
});

// ---------- room and role display ----------

function renderRoom() {
    $("roomStatus").textContent = state.peerSid
        ? `Room ${state.room}: 2 people`
        : `Room ${state.room}: waiting for the other person`;
    $("remoteNote").textContent = state.peerSid ? "Connecting video" : "Waiting for someone to join.";
    renderRoles();
}

function renderRoles() {
    const you = $("youRole"), other = $("otherRole");
    const meSign = iAmSigner();
    const otherSign = state.signMode && !meSign;
    you.textContent = meSign ? "Signing" : state.signMode ? "Speaking" : "Sign mode off";
    you.dataset.active = String(meSign);
    other.textContent = !state.peerSid ? "Not here" : otherSign ? "Signing" : state.signMode ? "Speaking" : "Sign mode off";
    other.dataset.active = String(Boolean(state.peerSid) && otherSign);

    const btn = $("signBtn");
    btn.setAttribute("aria-pressed", String(meSign));
    btn.textContent = meSign ? "Stop signing" : "I will sign";
    const noCamera = !state.stream || state.stream.getVideoTracks().length === 0;
    btn.disabled = !meSign && (noCamera || !state.modelReady);
    btn.title = btn.disabled ? (noCamera ? "Needs your camera" : "No trained model found on the server") : "";

    const mode = $("modeLine");
    mode.dataset.on = String(state.signMode);
    if (!state.signMode) {
        mode.textContent = noCamera
            ? "Sign mode is off. Your camera is not available, so you can read and listen but not sign."
            : "Sign mode is off. The person who signs presses \"I will sign\" on their own screen.";
    } else if (meSign) {
        mode.textContent = "You are signing. Your camera is read for letters A to J and the other person sees the word you build.";
    } else {
        mode.textContent = "The other person is signing. You will see their word, and what you say is matched to sign phrases for them.";
    }
    $("sendBtn").disabled = !meSign;
    $("backBtn").disabled = !meSign;
}

// ---------- sign mode ----------

$("signBtn").addEventListener("click", () => {
    socket.emit("toggle_sign_mode", { enabled: !iAmSigner() });
});

socket.on("sign_mode_changed", (data) => applySignMode(data.enabled, data.signer));

function applySignMode(enabled, signer) {
    state.signMode = enabled;
    state.signer = signer;
    stopHandTracking();
    stopSpeech();
    clearHandCanvas();
    setWord("", null, 0);
    if (enabled && signer === state.mySid) {
        startHandTracking();
    } else if (enabled) {
        startSpeech();
    } else {
        $("speechStatus").textContent = "Speech recognition starts on the screen of the person who is not signing.";
    }
    renderRoles();
}

$("sendBtn").addEventListener("click", () => socket.emit("commit_word"));
$("backBtn").addEventListener("click", () => socket.emit("backspace"));

function setWord(word, letter, confidence) {
    setText($("wordOut"), word, "Nothing yet");
    $("letterOut").textContent = letter || "none";
    $("confMeter").value = confidence;
    $("confOut").textContent = `${Math.round(confidence)}%`;
}

socket.on("sign_prediction", (d) => setWord(d.word, d.letter, d.confidence));

socket.on("word_committed", (d) => {
    setWord("", null, 0);
    if (!d.word) return;
    addLog("Signed", d.word);
    if (!iAmSigner() && $("speakAloud").checked) speakAloud(d.word);
});

function speakAloud(text) {
    if (!("speechSynthesis" in window)) {
        showAlert("tts", "This browser cannot read words aloud. The word is shown in the conversation list.", "info");
        return;
    }
    try {
        const u = new SpeechSynthesisUtterance(text);
        u.onerror = () => showAlert("tts", "The browser could not read the word aloud.", "info");
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(u);
    } catch (err) {
        showAlert("tts", "The browser could not read the word aloud.", "info");
    }
}

// ---------- hand tracking (signer only) ----------

let hands = null;
let trackingTimer = null;
let trackingBusy = false;

function clearHandCanvas() {
    const c = $("localHandCanvas");
    c.getContext("2d").clearRect(0, 0, c.width, c.height);
}

function startHandTracking() {
    if (typeof Hands === "undefined") {
        showAlert("hands", "The hand tracking files did not load, so signs cannot be read. Reload the page.");
        return;
    }
    clearAlert("hands");
    if (!hands) {
        hands = new Hands({ locateFile: (file) => `/static/vendor/mediapipe-hands/${file}` });
        hands.setOptions({ maxNumHands: 1, modelComplexity: 1, minDetectionConfidence: 0.7, minTrackingConfidence: 0.7 });
        hands.onResults(onHandResults);
    }
    const video = $("localVideo");
    trackingTimer = setInterval(async () => {
        if (trackingBusy || video.readyState < 2) return;
        trackingBusy = true;
        try {
            await hands.send({ image: video });
        } catch (err) {
            showAlert("hands", "Hand tracking stopped: " + err.message);
            stopHandTracking();
        } finally {
            trackingBusy = false;
        }
    }, 66);
}

function stopHandTracking() {
    if (trackingTimer) clearInterval(trackingTimer);
    trackingTimer = null;
}

const HAND_LINES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];

function onHandResults(results) {
    if (!iAmSigner()) return;
    const canvas = $("localHandCanvas");
    const video = $("localVideo");
    if (canvas.width !== video.videoWidth) { canvas.width = video.videoWidth; canvas.height = video.videoHeight; }
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const found = results.multiHandLandmarks && results.multiHandLandmarks[0];
    if (!found) {
        socket.emit("hand_lost");
        return;
    }
    ctx.strokeStyle = "#ffb27a";
    ctx.lineWidth = 4;
    ctx.beginPath();
    HAND_LINES.forEach(([a, b]) => {
        ctx.moveTo(found[a].x * canvas.width, found[a].y * canvas.height);
        ctx.lineTo(found[b].x * canvas.width, found[b].y * canvas.height);
    });
    ctx.stroke();
    // Raw MediaPipe coordinates from the unflipped frame. The server
    // normalises them with the same code that normalised the training data.
    socket.emit("hand_landmarks", { landmarks: found.map((p) => [p.x, p.y, p.z]) });
}

// ---------- speech recognition (the person who is not signing) ----------

let recognition = null;
let speechWanted = false;
let speechFailures = 0;

function startSpeech() {
    const API = window.SpeechRecognition || window.webkitSpeechRecognition;
    const status = $("speechStatus");
    const hasMic = state.stream && state.stream.getAudioTracks().length > 0;
    if (!API) {
        status.textContent = "This browser has no speech recognition, so what you say is not turned into signs. Chrome and Edge have it.";
        return;
    }
    if (!hasMic) {
        status.textContent = "No microphone, so what you say is not turned into signs.";
        return;
    }
    speechWanted = true;
    speechFailures = 0;
    status.textContent = "Listening. Say hello, help, no, thank you or yes to see it signed.";
    launchRecognition(API);
}

function launchRecognition(API) {
    recognition = new API();
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.lang = "en-US";
    recognition.onresult = (event) => {
        speechFailures = 0;
        const text = event.results[event.results.length - 1][0].transcript.trim();
        if (text) socket.emit("speech_text", { text });
    };
    recognition.onerror = (e) => {
        if (e.error === "not-allowed" || e.error === "service-not-allowed") {
            speechWanted = false;
            $("speechStatus").textContent = "The microphone or speech service is blocked, so speech is not recognised.";
            showAlert("speech", "Speech recognition was blocked. Allow the microphone in the address bar and press \"I will sign\" again.");
        } else if (e.error === "network") {
            $("speechStatus").textContent = "Speech recognition needs an internet connection in Chrome and Edge. It is not working right now.";
        } else if (e.error !== "no-speech" && e.error !== "aborted") {
            speechFailures += 1;
        }
    };
    recognition.onend = () => {
        if (!speechWanted) return;
        if (speechFailures >= 5) {
            $("speechStatus").textContent = "Speech recognition keeps stopping, so it is off.";
            speechWanted = false;
            return;
        }
        setTimeout(() => { if (speechWanted) { try { recognition.start(); } catch (err) { /* already running */ } } }, 400);
    };
    try { recognition.start(); } catch (err) { /* already running */ }
}

function stopSpeech() {
    speechWanted = false;
    if (recognition) {
        recognition.onend = null;
        try { recognition.stop(); } catch (err) { /* not running */ }
    }
    recognition = null;
}

socket.on("speech_caption", (d) => {
    setText($("heard"), d.text, "Nothing yet");
    addLog("Said", d.text);
});

// ---------- skeleton playback ----------

const skeletonCanvas = $("skeletonCanvas");
const sctx = skeletonCanvas.getContext("2d");
const FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
const ARM_LINES = [[11, 13], [13, 15], [12, 14], [14, 16], [11, 12]];
const ARM_POINTS = [11, 12, 13, 14, 15, 16];

let lastPhrases = [];
let playToken = 0;

socket.on("play_sign", (d) => {
    lastPhrases = d.phrases;
    $("replayBtn").disabled = false;
    if (reducedMotion) {
        $("skeletonLabel").textContent = `Sign ready: ${d.phrases.join(", ")}. Press Play again to watch it.`;
    } else {
        playPhrases(lastPhrases);
    }
});

$("replayBtn").addEventListener("click", () => playPhrases(lastPhrases));

function fillGaps(frames, key) {
    let last = null;
    const out = frames.map((f) => (last = f[key] || last));
    let next = null;
    for (let i = out.length - 1; i >= 0; i--) {
        if (out[i]) next = out[i]; else out[i] = next;
    }
    return out;
}

async function playPhrases(phrases) {
    const token = ++playToken;
    for (const phrase of phrases) {
        $("skeletonLabel").textContent = `Sign for: ${phrase}`;
        let data;
        try {
            const res = await fetch(`/skeleton_data/${encodeURIComponent(phrase)}`);
            if (!res.ok) throw new Error(res.status);
            data = await res.json();
        } catch (err) {
            showAlert("skeleton", `Could not load the sign for ${phrase}.`, "info");
            continue;
        }
        if (token !== playToken) return;
        await playOne(data, token);
        if (token !== playToken) return;
    }
}

function playOne(data, token) {
    const frames = data.frames;
    const tracks = {
        left: fillGaps(frames, "left_hand"),
        right: fillGaps(frames, "right_hand"),
    };
    const fps = data.fps || 20;
    return new Promise((resolve) => {
        const t0 = performance.now();
        function tick(now) {
            if (token !== playToken) return resolve();
            const i = Math.min(frames.length - 1, Math.floor(((now - t0) / 1000) * fps));
            drawSkeleton(frames[i], tracks.left[i], tracks.right[i]);
            if (i >= frames.length - 1) return setTimeout(resolve, 400);
            requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
    });
}

function drawLines(points, lines, w, h) {
    sctx.beginPath();
    lines.forEach(([a, b]) => {
        if (points[a] && points[b]) {
            sctx.moveTo(points[a][0] * w, points[a][1] * h);
            sctx.lineTo(points[b][0] * w, points[b][1] * h);
        }
    });
    sctx.stroke();
}

function drawDots(points, indexes, w, h, r) {
    indexes.forEach((i) => {
        if (!points[i]) return;
        sctx.beginPath();
        sctx.arc(points[i][0] * w, points[i][1] * h, r, 0, Math.PI * 2);
        sctx.fill();
    });
}

function drawSkeleton(frame, left, right) {
    const w = skeletonCanvas.width, h = skeletonCanvas.height;
    sctx.clearRect(0, 0, w, h);
    sctx.lineWidth = 3;
    sctx.lineCap = "round";
    if (frame.face) {
        sctx.strokeStyle = "#8f8f88";
        drawLines(frame.face, FACE_OVAL.map((v, k) => [v, FACE_OVAL[(k + 1) % FACE_OVAL.length]]), w, h);
    }
    if (frame.pose) {
        sctx.strokeStyle = sctx.fillStyle = "#c9c9c3";
        drawLines(frame.pose, ARM_LINES, w, h);
        drawDots(frame.pose, ARM_POINTS, w, h, 5);
    }
    [[left, "#ffb27a"], [right, "#ffffff"]].forEach(([pts, color]) => {
        if (!pts) return;
        sctx.strokeStyle = sctx.fillStyle = color;
        sctx.lineWidth = 4;
        drawLines(pts, HAND_LINES, w, h);
        drawDots(pts, pts.map((_, i) => i), w, h, 4);
    });
}
