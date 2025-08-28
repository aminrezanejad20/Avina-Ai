// server.js - پلتفرم هوش مصنوعی آوینا با RAG و استریم
require('dotenv').config();
const express = require('express');
const { OpenAI } = require('openai');
const fs = require('fs');
const path = require('path');
const pdfParse = require('pdf-parse');
const mammoth = require('mammoth');
const xlsx = require('xlsx');
const fileUpload = require('express-fileupload');
const { parse } = require('node-html-parser');
const fetch = require('node-fetch');

globalThis.fetch = fetch;

const app = express();
const PORT = process.env.PORT || 3000;

// --- تنظیمات OpenAI ---
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: 'https://api.avalai.ir/v1'
});

// --- مسیرها ---
const KNOWLEDGE_PATH = path.join(__dirname, 'knowledge', 'knowledge.json');
const PUBLIC_PATH = path.join(__dirname, 'public');

// --- میدلورها ---
app.use(fileUpload({
  limits: { fileSize: 50 * 1024 * 1024 },
  abortOnLimit: true
}));
app.use(express.static(PUBLIC_PATH));
app.use(express.json());

// --- ذخیره جلسات ---
const sessions = new Map();

// --- ایجاد knowledge.json ---
if (!fs.existsSync(path.dirname(KNOWLEDGE_PATH))) {
  fs.mkdirSync(path.dirname(KNOWLEDGE_PATH), { recursive: true });
}
if (!fs.existsSync(KNOWLEDGE_PATH)) {
  fs.writeFileSync(KNOWLEDGE_PATH, JSON.stringify({ chunks: [] }, null, 2), 'utf8');
}

// --- توابع پردازش ---
async function extractTextFromBuffer(buffer, filename) {
  const ext = path.extname(filename).toLowerCase();
  try {
    if (ext === '.pdf') {
      const data = await pdfParse(buffer);
      return data.text.trim();
    } else if (ext === '.docx') {
      const result = await mammoth.extractRawText({ buffer });
      return result.value.trim();
    } else if (ext === '.xlsx' || ext === '.xls') {
      const workbook = xlsx.read(buffer, { type: 'buffer' });
      let text = '';
      workbook.SheetNames.forEach(sheetName => {
        const sheet = xlsx.utils.sheet_to_json(workbook.Sheets[sheetName], { header: 1 });
        sheet.forEach(row => {
          text += row.filter(cell => cell).join(' ') + '\n';
        });
      });
      return text.trim();
    } else if (ext === '.txt') {
      return buffer.toString('utf8').trim();
    } else {
      throw new Error(`فرمت ${ext} پشتیبانی نمی‌شود`);
    }
  } catch (error) {
    throw new Error(`خطا در خواندن فایل: ${error.message}`);
  }
}

async function extractTextFromURL(url) {
  try {
    const res = await fetch(url, { headers: { 'User-Agent': 'AvinaAI/1.0' } });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const html = await res.text();
    const root = parse(html);
    root.querySelectorAll('script', 'style', 'nav', 'footer', 'header').forEach(el => el.remove());
    return root.text.replace(/\s+/g, ' ').trim() || 'متنی استخراج نشد.';
  } catch (error) {
    throw new Error(`خطا در کرال: ${error.message}`);
  }
}

function chunkText(text, chunkSize = 300, overlap = 30) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    let end = Math.min(start + chunkSize, text.length);
    chunks.push(text.slice(start, end));
    start = end - overlap;
  }
  return chunks;
}

function simpleEmbedding(text) {
  let hash = 0;
  for (let i = 0; i < Math.min(text.length, 100); i++) {
    const char = text.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return [hash % 1000, (hash * 7) % 1000, (hash * 13) % 1000];
}

function loadKnowledge() {
  try {
    return JSON.parse(fs.readFileSync(KNOWLEDGE_PATH, 'utf8')) || { chunks: [] };
  } catch (err) {
    console.error('Failed to load knowledge:', err);
    return { chunks: [] };
  }
}

function saveKnowledge(knowledge) {
  try {
    fs.writeFileSync(KNOWLEDGE_PATH, JSON.stringify(knowledge, null, 2), 'utf8');
  } catch (err) {
    console.error('Failed to save knowledge:', err);
  }
}

function cosineSimilarity(vecA, vecB) {
  const dot = vecA.reduce((s, a, i) => s + a * vecB[i], 0);
  const magA = Math.sqrt(vecA.reduce((s, a) => s + a * a, 0));
  const magB = Math.sqrt(vecB.reduce((s, b) => s + b * b, 0));
  return magA && magB ? dot / (magA * magB) : 0;
}

async function retrieveRelevantChunks(query, sessionId, topK = 3) {
  const knowledge = loadKnowledge();
  const queryEmbed = simpleEmbedding(query);
  const candidates = knowledge.chunks.filter(c => !sessionId || c.sessionId === sessionId);
  const scored = candidates.map(c => ({
    chunk: c,
    score: cosineSimilarity(queryEmbed, c.embedding)
  })).sort((a, b) => b.score - a.score);
  return scored.slice(0, topK).map(s => s.chunk.text);
}

// --- API ها ---
app.get('/api/models', async (req, res) => {
  try {
    const response = await fetch('https://api.avalai.ir/public/models');
    const data = await response.json();
    const models = Array.isArray(data.data) ? data.data : (Array.isArray(data) ? data : []);
    res.json(models.length ? models : [{ id: 'gpt-3.5-turbo' }, { id: 'gpt-4o-mini' }]);
  } catch {
    res.json([{ id: 'gpt-3.5-turbo' }, { id: 'gpt-4o-mini' }, { id: 'mistral' }]);
  }
});

app.post('/api/upload-knowledge', async (req, res) => {
  const sessionId = req.body.session_id || 'default';
  const MAX_TEXT_LENGTH = 100000;

  try {
    let text = '';
    let source = '';

    if (req.files?.file) {
      const file = req.files.file;
      source = file.name;
      text = await extractTextFromBuffer(file.data, file.name);
    } else if (req.body.url?.trim()) {
      source = req.body.url.trim();
      text = await extractTextFromURL(source);
    } else {
      return res.status(400).json({ error: 'فایل یا URL ارسال نشده.' });
    }

    if (!text) return res.status(400).json({ error: 'متنی استخراج نشد.' });
    if (text.length > MAX_TEXT_LENGTH) {
      return res.status(400).json({
        error: `متن خیلی طولانی است. حداکثر: ${MAX_TEXT_LENGTH} کاراکتر. متن شما: ${text.length} کاراکتر.`
      });
    }

    const chunks = chunkText(text);
    text = undefined;

    const knowledge = loadKnowledge();
    const newChunks = chunks.map(chunk => ({
      id: Date.now() + Math.random(),
      text: chunk,
      embedding: simpleEmbedding(chunk),
      sessionId,
      source,
      timestamp: new Date().toISOString()
    }));

    knowledge.chunks.push(...newChunks);
    saveKnowledge(knowledge);

    res.json({ message: `${newChunks.length} تکه از "${source}" پردازش شد.` });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// --- چت با استریم ---
app.post('/api/chat', async (req, res) => {
  try {
    const {
      model, systemPrompt, messages, temperature, max_tokens, top_p, stop,
      session_id, summarization
    } = req.body;

    if (!model || !messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: 'داده ناقص' });
    }

    const sid = session_id || 'default';
    let session = sessions.get(sid) || { history: [], summary: '' };
    let { history, summary } = session;

    // --- RAG ---
    const relevantContext = await retrieveRelevantChunks(messages[0].content, sid, 3);
    let contextText = '';
    if (relevantContext.length > 0) {
      contextText = '🔍 اطلاعات مرتبط از پایگاه دانش:\n' + relevantContext.join('\n---\n') + '\n\n';
    }

    // --- خلاصه‌سازی ---
    let needSummarize = false;
    if (summarization?.mode === "tokens") {
      const totalTokens = history.reduce((sum, m) => sum + m.content.length / 4, 0);
      if (totalTokens >= (summarization.tokenThreshold || 1500)) needSummarize = true;
    } else if (summarization?.mode === "messages") {
      if (history.length >= (summarization.messageThreshold || 6)) needSummarize = true;
    } else if (summarization?.mode === "hybrid") {
      const totalTokens = history.reduce((sum, m) => sum + m.content.length / 4, 0);
      if (history.length >= (summarization.messageThreshold || 6) || totalTokens >= (summarization.tokenThreshold || 1500)) {
        needSummarize = true;
      }
    }

    if (needSummarize) {
      const fullHistoryText = history.map(m => `${m.role}: ${m.content}`).join('\n');
      const prompt = `لطفاً این مکالمه را خلاصه کن: ${fullHistoryText}`;
      const summaryRes = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: prompt }],
        max_tokens: 300
      });
      summary = summaryRes.choices[0].message.content;
      history = history.slice(-3);
    }

    // --- آماده‌سازی پیام‌ها ---
    const messagesToSend = [
      { role: 'system', content: systemPrompt || 'تو یک دستیار هوشمند هستی.' }
    ];

    if (summary) {
      messagesToSend.push({ role: 'system', content: `خلاصه مکالمه قبلی: ${summary}` });
    }
    if (contextText) {
      messagesToSend.push({ role: 'system', content: contextText });
    }
    messagesToSend.push(...history, ...messages);

    // --- استریم پاسخ ---
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    const stream = await openai.chat.completions.create({
      model,
      messages: messagesToSend,
      temperature: temperature ?? 0.7,
      max_tokens: max_tokens ?? 1024,
      top_p: top_p ?? 0.9,
      stop: stop?.length > 0 ? stop.filter(s => s.trim()) : undefined,
      stream: true
    });

    let fullReply = '';
    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || '';
      fullReply += content;
      res.write(`data: ${JSON.stringify({ content })}\n\n`);
    }

    res.write('data: [DONE]\n\n');
    res.end();

    // --- به‌روزرسانی تاریخچه ---
    history.push({ role: 'user', content: messages[0].content });
    history.push({ role: 'assistant', content: fullReply });
    sessions.set(sid, { history, summary });

  } catch (error) {
    res.write(`data: ${JSON.stringify({ content: '❌ خطایی رخ داد: ' + (error.message || 'ناشناخته') })}\n\n`);
    res.write('data: [DONE]\n\n');
    res.end();
  }
});

// --- شروع سرور ---
app.listen(PORT, () => {
  console.log(`Server is running: http://localhost:${PORT}`);
  console.log(`Knowledge Base: ${KNOWLEDGE_PATH}`);
});