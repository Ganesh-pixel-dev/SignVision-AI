/*
 call.js
 --------
 Ties together:
 1. WebRTC peer connection (video/audio call between two browser tabs)
 2. Socket.IO signaling (offer/answer/ICE relay + sign-mode toggle broadcast)
 3. Client-side MediaPipe Hands (for the deaf participant's outgoing hand
    landmarks -> sent to the Python backend for classification)
 4. Browser Web Speech API (for the hearing participant's outgoing speech
    -> sent to the backend to match against the sign-phrase library)
 5. A canvas-based skeleton player that animates a recorded phrase's
    landmark sequence when the backend says "play_sign"
*/

const socket = io();
let room = null;
let pc = null;
let localStream = null;
let signModeEnabled = false;

const localVideo = document.getElementById("localVideo");
const remoteVideo = document.getElementById("remoteVideo");
const localCaption = document.getElementById("localCaption");
const remoteCaption = document.getElementById("remoteCaption");
const skeletonCanvas = document.getElementById("skeletonCanvas");
const skeletonCtx = skeletonCanvas.getContext("2d");
const localHandCanvas = document.getElementById("localHandCanvas");
const localHandCtx = localHandCanvas.getContext("2d");

const rtcConfig = {
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
};

// ---------------- Join / WebRTC setup ----------------

document.getElementById("joinBtn").addEventListener("click", async () => {
    room = document.getElementById("roomInput").value.trim() || "demo-room";
    await startLocalMedia();
    socket.emit("join", { room });
    setupPeerConnection();
});

async function startLocalMedia() {
    localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    localVideo.srcObject = localStream;
    localHandCanvas.width = 480;
    localHandCanvas.height = 480;
}

function setupPeerConnection() {
    pc = new RTCPeerConnection(rtcConfig);

    localStream.getTracks().forEach((track) => pc.addTrack(track, localStream));

    pc.ontrack = (event) => {
        remoteVideo.srcObject = event.streams[0];
    };

    pc.onicecandidate = (event) => {
        if (event.candidate) {
            socket.emit("signal", { room, type: "candidate", candidate: event.candidate });
        }
    };
}

socket.on("peer_joined", async () => {
    // The peer who was already in the room initiates the offer
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    socket.emit("signal", { room, type: "offer", sdp: offer });
});

socket.on("signal", async (data) => {
    if (data.type === "offer") {
        await pc.setRemoteDescription(new RTCSessionDescription(data.sdp));
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);
        socket.emit("signal", { room, type: "answer", sdp: answer });
    } else if (data.type === "answer") {
        await pc.setRemoteDescription(new RTCSessionDescription(data.sdp));
    } else if (data.type === "candidate") {
        try {
            await pc.addIceCandidate(new RTCIceCandidate(data.candidate));
        } catch (e) {
            console.error("Error adding ICE candidate", e);
        }
    }
});

// ---------------- Sign mode toggle ----------------

const signModeToggle = document.getElementById("signModeToggle");
signModeToggle.addEventListener("change", () => {
    signModeEnabled = signModeToggle.checked;
    socket.emit("toggle_sign_mode", { room, enabled: signModeEnabled });
    applySignMode(signModeEnabled);
});

socket.on("sign_mode_changed", (data) => {
    signModeToggle.checked = data.enabled;
    signModeEnabled = data.enabled;
    applySignMode(data.enabled);
});

function applySignMode(enabled) {
    if (enabled) {
        startHandTracking();
        startSpeechRecognition();
    } else {
        stopHandTracking();
        stopSpeechRecognition();
        skeletonCtx.clearRect(0, 0, skeletonCanvas.width, skeletonCanvas.height);
        localHandCtx.clearRect(0, 0, localHandCanvas.width, localHandCanvas.height);
    }
}

// ---------------- MediaPipe Hands (client-side landmark extraction) ----------------

let handsModel = null;
let handCameraLoop = null;

function startHandTracking() {
    if (!handsModel) {
        handsModel = new Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
        });
        handsModel.setOptions({
            maxNumHands: 1,
            modelComplexity: 1,
            minDetectionConfidence: 0.7,
            minTrackingConfidence: 0.7,
        });
        handsModel.onResults(onHandResults);
    }

    handCameraLoop = setInterval(async () => {
        if (localVideo.readyState >= 2) {
            await handsModel.send({ image: localVideo });
        }
    }, 100); // ~10 fps is plenty for gesture holding
}

function stopHandTracking() {
    if (handCameraLoop) clearInterval(handCameraLoop);
}

function onHandResults(results) {
    localHandCtx.clearRect(0, 0, localHandCanvas.width, localHandCanvas.height);

    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
        return;
    }

    const landmarks = results.multiHandLandmarks[0];

    // draw for local feedback
    drawConnectors(landmarks);

    // Normalize EXACTLY like core/hand_tracker.py: wrist-relative, scaled by
    // distance from wrist (0) to middle-finger MCP (9)
    const wrist = landmarks[0];
    const points = landmarks.map((lm) => [lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]);
    const mcp = points[9];
    let scale = Math.sqrt(mcp[0] ** 2 + mcp[1] ** 2 + mcp[2] ** 2);
    if (scale < 1e-6) scale = 1e-6;
    const features = points.flatMap((p) => [p[0] / scale, p[1] / scale, p[2] / scale]);

    socket.emit("hand_landmarks", { room, features });
}

function drawConnectors(landmarks) {
    const HAND_CONNECTIONS = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [5, 9], [9, 10], [10, 11], [11, 12],
        [9, 13], [13, 14], [14, 15], [15, 16],
        [13, 17], [17, 18], [18, 19], [19, 20],
        [0, 17],
    ];
    const w = localHandCanvas.width, h = localHandCanvas.height;
    localHandCtx.strokeStyle = "#4fd1c5";
    localHandCtx.lineWidth = 2;
    HAND_CONNECTIONS.forEach(([a, b]) => {
        localHandCtx.beginPath();
        localHandCtx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
        localHandCtx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
        localHandCtx.stroke();
    });
}

socket.on("sign_prediction", (data) => {
    localCaption.textContent = data.word
        ? `${data.word}  (current: ${data.letter || ""} ${data.confidence}%)`
        : "\u00A0";
    remoteCaption.textContent = localCaption.textContent;
});

socket.on("word_committed", (data) => {
    remoteCaption.textContent = data.word ? `Signed: "${data.word}"` : "\u00A0";
    if (data.word && "speechSynthesis" in window) {
        const utter = new SpeechSynthesisUtterance(data.word);
        window.speechSynthesis.speak(utter);
    }
});

// ---------------- Web Speech API (hearing participant's speech -> text) ----------------

let recognition = null;

function startSpeechRecognition() {
    const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognitionAPI) {
        console.warn("Web Speech API not supported in this browser (try Chrome).");
        return;
    }
    recognition = new SpeechRecognitionAPI();
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
        const text = event.results[event.results.length - 1][0].transcript;
        socket.emit("speech_text", { room, text });
    };

    recognition.onerror = (e) => console.warn("Speech recognition error:", e.error);
    recognition.onend = () => {
        if (signModeEnabled) recognition.start(); // keep listening while sign mode is on
    };

    recognition.start();
}

function stopSpeechRecognition() {
    if (recognition) {
        recognition.onend = null;
        recognition.stop();
    }
}

socket.on("speech_caption", (data) => {
    localCaption.textContent = data.text || "\u00A0";
});

// ---------------- Skeleton playback (text -> sign animation) ----------------

const HAND_CONNECTIONS_SKEL = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [0, 17],
];
const POSE_CONNECTIONS_SKEL = [[11, 13], [13, 15], [12, 14], [14, 16], [11, 12]];
const FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];

socket.on("play_sign", async (data) => {
    const res = await fetch(`/skeleton_data/${data.phrase}`);
    if (!res.ok) return;
    const skelData = await res.json();
    playSkeletonAnimation(skelData);
});

function playSkeletonAnimation(skelData) {
    const w = skeletonCanvas.width, h = skeletonCanvas.height;
    const frames = skelData.frames;
    const fps = skelData.fps || 30;
    let i = 0;

    if (window._skelInterval) clearInterval(window._skelInterval);

    window._skelInterval = setInterval(() => {
        if (i >= frames.length) {
            clearInterval(window._skelInterval);
            return;
        }
        const fr = frames[i];
        skeletonCtx.clearRect(0, 0, w, h);
        skeletonCtx.fillStyle = "rgba(20,22,26,0.0)";

        drawPointsSkel(fr.face, FACE_OVAL.map((idx, k) => [idx, FACE_OVAL[(k + 1) % FACE_OVAL.length]]), "#dc3232", w, h, true);
        drawPointsSkel(fr.pose, POSE_CONNECTIONS_SKEL, "#dc3232", w, h);
        drawPointsSkel(fr.left_hand, HAND_CONNECTIONS_SKEL, "#ffa500", w, h);
        drawPointsSkel(fr.right_hand, HAND_CONNECTIONS_SKEL, "#3399ff", w, h);

        i++;
    }, 1000 / fps);
}

function drawPointsSkel(points, connections, color, w, h, closedLoop) {
    if (!points) return;
    skeletonCtx.strokeStyle = color;
    skeletonCtx.fillStyle = color;
    skeletonCtx.lineWidth = 2;
    connections.forEach(([a, b]) => {
        if (points[a] && points[b]) {
            skeletonCtx.beginPath();
            skeletonCtx.moveTo(points[a][0] * w, points[a][1] * h);
            skeletonCtx.lineTo(points[b][0] * w, points[b][1] * h);
            skeletonCtx.stroke();
        }
    });
    points.forEach((p) => {
        skeletonCtx.beginPath();
        skeletonCtx.arc(p[0] * w, p[1] * h, 3, 0, Math.PI * 2);
        skeletonCtx.fill();
    });
}
