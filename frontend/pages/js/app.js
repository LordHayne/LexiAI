let mediaRecorder;
let audioChunks = [];

const recordBtn = document.getElementById("recordBtn");
const statusEl = document.getElementById("status");
const responseAudio = document.getElementById("responseAudio");

recordBtn.addEventListener("click", () => {
  if (recordBtn.textContent === "Start Aufnahme") {
    startRecording();
  } else {
    stopRecording();
  }
});

function startRecording() {
  navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    mediaRecorder.start();

    mediaRecorder.addEventListener("dataavailable", event => {
      audioChunks.push(event.data);
    });

    mediaRecorder.addEventListener("stop", () => {
      const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
      sendToLexi(audioBlob);
    });

    recordBtn.textContent = "Stop Aufnahme";
    updateStatus("🎤 Aufnahme läuft...", "success");
  }).catch(err => {
    updateStatus("❌ Fehler beim Starten der Aufnahme", "error");
    console.error(err);
  });
}

function stopRecording() {
  mediaRecorder.stop();
  recordBtn.textContent = "Start Aufnahme";
  updateStatus("🛑 Aufnahme gestoppt. Sende an Lexi…", "warning");
}

function sendToLexi(audioBlob) {
  const formData = new FormData();
  formData.append("file", audioBlob, "audio.webm");

  updateStatus("🚀 Sende an Lexi...", "warning");

  fetch("http://localhost:8000/api/audio", {  // Hier DEIN-LEXI-SERVER einsetzen!
    method: "POST",
    body: formData
  })
  .then(response => {
    if (!response.ok) {
      throw new Error(`Serverfehler: ${response.status}`);
    }
    return response.blob();
  })
  .then(blob => {
    const audioUrl = URL.createObjectURL(blob);
    responseAudio.src = audioUrl;
    responseAudio.style.display = "block"; // Audio sichtbar machen
    updateStatus("✅ Antwort von Lexi empfangen!", "success");
  })
  .catch(err => {
    updateStatus(`❌ Fehler: ${err.message}`, "error");
    console.error(err);
  });
}

function updateStatus(message, type) {
  statusEl.textContent = message;
  statusEl.className = `status ${type}`;
  statusEl.style.display = "block";
}
