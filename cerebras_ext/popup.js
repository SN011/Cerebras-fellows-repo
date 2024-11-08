/* popup.js */
let isRecording = false;
let isHumanToolRequest = false;

let socket = io('http://localhost:5000');

socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('tts_complete', (data) => {
    console.log('TTS synthesis complete:', data.message);
    const audio = document.getElementById('audio');
    audio.src = 'http://localhost:5000/get_audio?' + new Date().getTime();
    audio.style.display = 'block';
    audio.play().catch(error => {
        console.error('Error playing audio:', error);
    });
});

socket.on('new_message', (data) => {
    displayMessage(data.message, data.sender);
});

socket.on('request_human_input', () => {
    console.log('Server is requesting human input');
    document.getElementById('textInput').disabled = false;
    document.getElementById('sendTextButton').disabled = false;
    isHumanToolRequest = true;
});

socket.on('finished_chain', () => {
    console.log('Chain execution finished');
    document.getElementById('textInput').disabled = false;
    document.getElementById('sendTextButton').disabled = false;
    isHumanToolRequest = false;
});

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('sendTextButton').addEventListener('click', handleTextInput);
    document.getElementById('textInput').addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            handleTextInput();
        }
    });

    document.addEventListener('keydown', (event) => {
        if (event.key === 'p' || event.key === 'P') {
            const activeElement = document.activeElement;
            if (activeElement && activeElement.id !== 'textInput') {
                if (!isRecording) {
                    startRecording();
                } else {
                    stopRecording();
                }
            }
        }
    });

    document.getElementById('talkButton').addEventListener('click', () => {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    });
});

function handleTextInput() {
    if (isHumanToolRequest) {
        sendHumanInput();
    } else {
        sendTextInput();
    }
}

function sendTextInput() {
    const inputText = document.getElementById('textInput').value;
    if (!inputText.trim()) return;

    displayMessage(inputText, 'user');
    
    fetch('http://localhost:5000/text_input', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: inputText })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Synthesis request sent:', data);
    })
    .catch(error => {
        console.error('Error:', error);
    });

    document.getElementById('textInput').value = '';
}

function sendHumanInput() {
    const inputText = document.getElementById('textInput').value;
    if (!inputText.trim()) return;

    displayMessage(inputText, 'user');
    socket.emit('provide_human_input', { text: inputText });

    document.getElementById('textInput').value = '';
    document.getElementById('textInput').disabled = true;
    document.getElementById('sendTextButton').disabled = true;
    isHumanToolRequest = false;
}

function displayMessage(message, sender) {
    const chatWindow = document.getElementById('chatWindow');
    const messageElement = document.createElement('div');
    messageElement.className = `message ${sender}`;
    messageElement.textContent = message;
    chatWindow.appendChild(messageElement);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function startRecording() {
    console.log('Start recording');
    isRecording = true;

    const orb = document.querySelector('.orb');
    orb.classList.add('floating');

    fetch('http://localhost:5000/start_recording', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            console.log(data.status);
        });
}

function stopRecording() {
    console.log('Stop recording');
    isRecording = false;

    const orb = document.querySelector('.orb');
    orb.classList.remove('floating');

    fetch('http://localhost:5000/stop_recording', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            console.log(data.status);
            fetchAudioResponse();
        });
}

function fetchAudioResponse() {
    fetch('http://localhost:5000/talk', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            document.getElementById('response').innerText = data.response;
            
            const audio = document.getElementById('audio');
            audio.src = 'http://localhost:5000/get_audio?' + new Date().getTime();
            audio.style.display = 'block';
            audio.play().catch(error => {
                console.error('Error playing audio:', error);
            });
        })
        .catch(error => {
            console.error('Error:', error);
        });
}