"use strict";

const $ = (id) => document.getElementById(id);

function setAlert(message) {
    $("alert").hidden = !message;
    $("alert").textContent = message || "";
}

function setStatus(text, ok) {
    $("status").textContent = text;
    $("status").dataset.ok = String(ok);
}

async function poll() {
    try {
        const d = await (await fetch("/state")).json();
        $("word").textContent = d.word || "Nothing yet";
        $("letter").textContent = d.letter || "none";
        $("conf").value = d.confidence;
        $("confText").textContent = `${Math.round(d.confidence)}%`;
        $("history").replaceChildren(...[...d.history].reverse().map((w) => {
            const li = document.createElement("li");
            li.textContent = w;
            return li;
        }));
        if (d.camera_ok === false) {
            setStatus("No camera", false);
            setAlert("The server could not open the camera. Close other apps using it, or set CAMERA_INDEX, then restart.");
        } else if (!d.model_ready) {
            setStatus("No model", false);
            setAlert("No trained model found. Run training/train_model.py, then restart.");
        } else {
            setStatus(d.camera_ok ? "Running" : "Starting camera", Boolean(d.camera_ok));
            setAlert("");
        }
    } catch (err) {
        setStatus("Server not reachable", false);
    }
}

async function post(path) {
    try {
        const res = await fetch(path, { method: "POST" });
        const d = await res.json();
        if (d.error) setAlert(d.error);
    } catch (err) {
        setAlert("Could not reach the server.");
    }
    poll();
}

$("speakBtn").addEventListener("click", () => post("/speak"));
$("resetBtn").addEventListener("click", () => post("/reset"));
$("backBtn").addEventListener("click", () => post("/backspace"));

setInterval(poll, 300);
poll();
