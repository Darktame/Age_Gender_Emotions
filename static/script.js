const video = document.getElementById('video');
const resultsDiv = document.getElementById('results');

// Ask for webcam access
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => { video.srcObject = stream; })
    .catch(err => { console.error("Error accessing webcam:", err); });

function sendFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const dataUrl = canvas.toDataURL('image/jpeg');

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl })
    })
    .then(response => response.json())
    .then(data => {
        if (data.results) {
            resultsDiv.innerHTML = "";
            data.results.forEach((person, i) => {
                resultsDiv.innerHTML += `Person ${i+1}: Age ${person.age}, ${person.gender}, Emotion: ${person.emotion}<br>`;
            });
        } else if (data.error) {
            console.error("Error from server:", data.error);
        }
    })
    .catch(err => console.error("Request error:", err));
}

// Send frame every 500ms
setInterval(sendFrame, 500);
