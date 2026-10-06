const detectedLetter = document.getElementById("detectedLetter");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceText = document.getElementById("confidenceText");
const currentWord = document.getElementById("currentWord");
const historyList = document.getElementById("historyList");
const modelStatus = document.getElementById("modelStatus");

async function pollState() {
    try {
        const res = await fetch("/state");
        const data = await res.json();

        detectedLetter.textContent = data.letter || "-";
        confidenceFill.style.width = `${data.confidence}%`;
        confidenceText.textContent = `${data.confidence}%`;
        currentWord.textContent = data.word || "\u00A0";

        historyList.innerHTML = "";
        [...data.history].reverse().forEach((word) => {
            const li = document.createElement("li");
            li.textContent = word;
            historyList.appendChild(li);
        });

        modelStatus.textContent = data.model_ready
            ? "● Model loaded"
            : "⚠ No trained model - run training/train_model.py";
        modelStatus.style.color = data.model_ready ? "#4fd1c5" : "#ff8a3d";
    } catch (err) {
        modelStatus.textContent = "⚠ Connection lost";
        modelStatus.style.color = "#ff5c5c";
    }
}

document.getElementById("speakBtn").addEventListener("click", async () => {
    await fetch("/speak", { method: "POST" });
});

document.getElementById("resetBtn").addEventListener("click", async () => {
    await fetch("/reset", { method: "POST" });
});

document.getElementById("backspaceBtn").addEventListener("click", async () => {
    await fetch("/backspace", { method: "POST" });
});

setInterval(pollState, 300);
pollState();
