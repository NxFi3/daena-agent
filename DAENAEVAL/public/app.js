// Helper to show status messages
function setStatus(message, type = 'info') {
  const statusEl = document.getElementById('status');
  statusEl.textContent = message;
  statusEl.className = type;
}

// Fetch notes from server
async function fetchNotes(query = '') {
  try {
    const url = query
      ? `/api/search?q=${encodeURIComponent(query)}`
      : '/api/notes';
    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed to fetch notes');
    const notes = await res.json();
    renderNotes(notes);
  } catch (err) {
    setStatus(err.message, 'error');
  }
}

// Render notes list
function renderNotes(notes) {
  const list = document.getElementById('notes-list');
  list.innerHTML = '';
  if (notes.length === 0) {
    list.innerHTML = '<li>No notes found.</li>';
    return;
  }
  notes.forEach((note) => {
    const li = document.createElement('li');
    li.className = 'note-item';
    li.dataset.id = note.id;
    li.innerHTML = `
      <div class="note-header">
        <span class="note-title">${note.title}</span>
        <div class="note-actions">
          <button class="edit">Edit</button>
          <button class="delete">Delete</button>
        </div>
      </div>
      <div class="note-content">${note.content}</div>
      <small>Last modified: ${new Date(note.lastModified).toLocaleString()}</small>
    `;
    list.appendChild(li);
  });
}

// Add or update note
async function saveNote() {
  const titleEl = document.getElementById('title');
  const contentEl = document.getElementById('content');
  const title = titleEl.value.trim();
  const content = contentEl.value.trim();
  if (!title || !content) {
    setStatus('Title and content cannot be empty', 'error');
    return;
  }
  const payload = { title, content };
  const method = 'POST';
  const url = '/api/notes';
  try {
    const res = await fetch(url, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error('Failed to save note');
    setStatus('Note saved', 'info');
    titleEl.value = '';
    contentEl.value = '';
    fetchNotes();
  } catch (err) {
    setStatus(err.message, 'error');
  }
}

// Delete note
async function deleteNote(id) {
  if (!confirm('Delete this note?')) return;
  try {
    const res = await fetch(`/api/notes/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete note');
    setStatus('Note deleted', 'info');
    fetchNotes();
  } catch (err) {
    setStatus(err.message, 'error');
  }
}

// Edit note (populate form)
async function editNote(id) {
  try {
    const res = await fetch(`/api/notes`);
    if (!res.ok) throw new Error('Failed to fetch notes');
    const notes = await res.json();
    const note = notes.find((n) => n.id === id);
    if (!note) throw new Error('Note not found');
    document.getElementById('title').value = note.title;
    document.getElementById('content').value = note.content;
    // For simplicity, we just delete the old note after editing
    await deleteNote(id);
  } catch (err) {
    setStatus(err.message, 'error');
  }
}

// Event listeners
document.getElementById('save').addEventListener('click', saveNote);
document.getElementById('search').addEventListener('input', (e) => {
  const query = e.target.value.trim();
  fetchNotes(query);
});
document.getElementById('notes-list').addEventListener('click', (e) => {
  const target = e.target;
  if (target.classList.contains('delete')) {
    const id = target.closest('.note-item').dataset.id;
    deleteNote(id);
  } else if (target.classList.contains('edit')) {
    const id = target.closest('.note-item').dataset.id;
    editNote(id);
  }
});

// Initial load
fetchNotes();
