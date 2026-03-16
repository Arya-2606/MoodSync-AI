const API = "http://127.0.0.1:8000";
const audio = document.getElementById("audioPlayer");

// Load songs
window.onload = async () => {
    const res = await fetch(`${API}/songs`);
    const data = await res.json();

    const select = document.getElementById("songSelect");
    select.innerHTML = `<option value="">Select a song</option>`;

    data.songs.forEach(song => {
        const opt = document.createElement("option");
        opt.value = song;
        opt.textContent = song.replace(".mp3", "");
        select.appendChild(opt);
    });
};

// Play selected song
document.getElementById("songSelect").addEventListener("change", e => {
    const song = e.target.value;
    if (song) {
        audio.src = `${API}/songs/${song}`;
        audio.play();
    }
});

// Recommend
async function getRecommendations() {
    const song = document.getElementById("songSelect").value;
    if (!song) return alert("Select a song");

    const res = await fetch(`${API}/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ song })
    });

    const data = await res.json();
    renderResults(data.recommendations);
}

// Render cards
function renderResults(recs) {
    const div = document.getElementById("results");
    div.innerHTML = "";

    recs.forEach(r => {
        const percent = Math.min(r.similarity * 100, 100);

        div.innerHTML += `
      <div class="card" onclick="playSong('${r.name}')">
        <b>${r.name}</b><br>
        Mood: ${r.mood}<br>
        Similarity: ${r.similarity}
        <div class="bar-bg">
          <div class="bar" style="width:${percent}%"></div>
        </div>
      </div>
    `;
    });
}

// Play recommended song
function playSong(song) {
    audio.src = `${API}/songs/${song}`;
    audio.play();
}
