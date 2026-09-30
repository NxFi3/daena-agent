const express = require('express');
const { v4: uuidv4 } = require('uuid');
const low = require('lowdb');
const FileSync = require('lowdb/adapters/FileSync');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;

// Ensure data directory exists
const dataDir = path.join(__dirname, 'data');
if (!fs.existsSync(dataDir)) {
  fs.mkdirSync(dataDir);
}

const file = path.join(dataDir, 'notes.json');
const adapter = new FileSync(file);
const db = low(adapter);

// Initialize database with default structure if file does not exist
function initDB() {
  db.defaults({ notes: [] }).write();
}

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Helper to get current timestamp
function now() {
  return new Date().toISOString();
}

// CRUD Endpoints
app.get('/api/notes', (req, res) => {
  res.json(db.get('notes').value());
});

app.post('/api/notes', (req, res) => {
  const { title, content } = req.body;
  if (!title || !content) {
    return res.status(400).json({ error: 'Title and content required' });
  }
  const note = {
    id: uuidv4(),
    title,
    content,
    lastModified: now(),
  };
  db.get('notes').push(note).write();
  res.status(201).json(note);
});

app.put('/api/notes/:id', (req, res) => {
  const { id } = req.params;
  const { title, content } = req.body;
  const note = db.get('notes').find({ id }).value();
  if (!note) {
    return res.status(404).json({ error: 'Note not found' });
  }
  if (title !== undefined) note.title = title;
  if (content !== undefined) note.content = content;
  note.lastModified = now();
  db.get('notes').find({ id }).assign(note).write();
  res.json(note);
});

app.delete('/api/notes/:id', (req, res) => {
  const { id } = req.params;
  const removed = db.get('notes').remove({ id }).write();
  if (removed.length === 0) {
    return res.status(404).json({ error: 'Note not found' });
  }
  res.status(204).end();
});

// Search endpoint
app.get('/api/search', (req, res) => {
  const q = (req.query.q || '').toLowerCase();
  const results = db.get('notes')
    .filter((n) => n.title.toLowerCase().includes(q) || n.content.toLowerCase().includes(q))
    .value();
  res.json(results);
});

// Initialize DB and start server
initDB();
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});

